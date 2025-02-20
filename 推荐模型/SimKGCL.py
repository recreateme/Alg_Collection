import torch
import scipy.sparse as sp
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F
import numpy as np
from recbole.model.layers import SparseDropout


class SimKGCL(KnowledgeRecommender):
    """
    SimKGCL（基于知识图增强的协同学习相似性感知模型）用于推荐系统。
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimKGCL, self).__init__(config, dataset)

        # 加载数据集信息
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._user = dataset.inter_feat[dataset.uid_field]  # 从交互数据中获取用户ID数组
        self._item = dataset.inter_feat[dataset.iid_field]  # 从交互数据中获取物品ID数组

        # 加载配置参数
        self.latent_dim = config['embedding_size']  # 嵌入维度
        self.n_layers = config['n_layers']  # 传播层数
        self.reg_weight = config['reg_weight']  # 正则化权重
        self.require_pow = config['require_pow']  # 是否需要幂归一化
        self.layer_cl = config['layer_cl']  # 对比学习的层
        self.cl_rate = config['cl_rate']  # 对比学习损失的比例
        self.tau = config['tau']  # softmax温度参数
        self.kg_drop_rate = config['kg_drop_rate']  # 知识图谱边的丢弃率
        self.ig_drop_rate = config['ig_drop_rate']  # 交互图边的丢弃率
        self.mess_drop_rate = config['mess_drop_rate']  # 消息传递的丢弃率

        # 定义层和嵌入
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.entity_embedding = torch.nn.Embedding(self.n_entities, self.latent_dim)
        self.relation_embedding = torch.nn.Embedding(self.n_relations + 1, self.latent_dim)

        self.message_drop = torch.nn.Dropout(self.mess_drop_rate)  # 消息聚合时的丢弃
        self.node_drop = SparseDropout(self.ig_drop_rate)  # 交互图的节点丢弃

        # 定义损失函数
        self.mf_loss = BPRLoss()  # BPR（贝叶斯个性化排序）损失
        self.reg_loss = EmbLoss()  # 嵌入的正则化损失

        # 用于完整排序评估加速的存储
        self.restore_user_e = None
        self.restore_item_e = None

        # 生成中间数据结构
        self.norm_adj_matrix, self.user_item_matrix = self.get_norm_adj_mat()
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        self.all_hs = torch.LongTensor(self.kg_graph.row).to(self.device)  # 头实体
        self.all_ts = torch.LongTensor(self.kg_graph.col).to(self.device)  # 尾实体
        self.all_rs = torch.LongTensor(self.kg_graph.data).to(self.device)  # 关系

        # 初始化模型参数
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self):
        """
        获取用户和物品的标准化交互矩阵。

        构建从训练数据中得来的方阵，并使用拉普拉斯矩阵进行标准化。

        返回：
            - 标准化后的交互矩阵的稀疏张量。
        """
        # 构建邻接矩阵
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # 标准化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7  # 添加epsilon避免除以零
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # 转换为稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        # 用户-物品交互矩阵
        L_ = L.tocsr()[: self.n_users, self.n_users:].tocoo()
        i_ = torch.LongTensor(np.array([L_.row, L_.col]))
        data_ = torch.FloatTensor(L_.data)
        SparseL_ = torch.sparse.FloatTensor(i_, data_, torch.Size(L_.shape))

        return SparseL.to(self.device), SparseL_.to(self.device)

    def get_ego_embeddings(self):
        """
        获取用户和物品/实体的嵌入。

        返回：
            - 组合后的用户和物品嵌入张量。
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def kg_agg(self, entity_emb, user_emb, relation_emb, all_h, all_t, all_r, inter_matrix, attention=True):
        """
        从知识图中聚合信息。

        参数：
            - entity_emb: 实体的嵌入
            - user_emb: 用户的嵌入
            - relation_emb: 关系的嵌入
            - all_h, all_t, all_r: KG三元组的头、尾和关系索引
            - inter_matrix: 用户-物品交互矩阵
            - attention: 是否使用注意力机制进行聚合

        返回：
            - 更新后的实体和用户嵌入。
        """
        from torch_scatter import scatter_softmax, scatter_mean

        n_entities = entity_emb.shape[0]
        edge_relation_emb = relation_emb[all_r]
        neigh_relation_emb = (entity_emb[all_t] * edge_relation_emb)

        if attention:
            # 计算注意力权重
            neigh_relation_emb_weight = self.calculate_sim_hrt(
                entity_emb[all_h], entity_emb[all_t], edge_relation_emb
            )
            neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
                                                                         neigh_relation_emb.shape[1])
            neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=all_h, dim=0)
            neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb)

        # 根据传入的关系聚合实体嵌入
        entity_agg = scatter_mean(src=neigh_relation_emb, index=all_h, dim_size=n_entities, dim=0)

        # 根据物品交互聚合用户嵌入
        user_agg = torch.sparse.mm(inter_matrix, entity_emb[:self.n_items]) #(m+n,m+n)()

        # 使用关系注意力计算用户聚合
        score = torch.mm(user_emb, relation_emb.t())    # (m,d)(d,r)---->(m,r)
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, relation_emb)) * user_agg

        return entity_agg, user_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        """
        计算头、关系和尾之间的相似度，用于注意力机制。
        注意：此方法遵循原始作者的实现，与论文描述略有不同。
        """
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(
            head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)
        ).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

    def kg_forward(self, ego_embeddings, Drop=False):
        """
        通过模型的知识图部分进行前向传播。

        参数：
            - ego_embeddings: 用户和实体的初始嵌入。
            - Drop: 是否应用丢弃。

        返回：
            - KG聚合后的更新嵌入。
        """
        user_emb, entity_emb = torch.split(ego_embeddings, [self.n_users, self.n_entities])

        # 对KG进行边采样以应用丢弃
        if Drop and self.kg_drop_rate > 0.0:
            all_h, all_t, all_r = self.edge_sampling(self.all_hs, self.all_ts, self.all_rs, 1 - self.kg_drop_rate)
        else:
            all_h, all_t, all_r = self.all_hs, self.all_ts, self.all_rs

        # 对交互图进行节点采样
        if Drop and self.ig_drop_rate > 0.0:
            inter_matrix = self.node_drop(self.user_item_matrix)
        else:
            inter_matrix = self.user_item_matrix

        relation_emb = self.relation_embedding.weight

        entity_emb, user_emb = self.kg_agg(entity_emb, user_emb, relation_emb, all_h, all_t, all_r, inter_matrix)

        # 如果启用，应用消息丢弃
        if Drop and self.mess_drop_rate > 0.0:
            entity_emb = self.message_drop(entity_emb)
            user_emb = self.message_drop(user_emb)

        # 归一化嵌入
        entity_emb = F.normalize(entity_emb)
        user_emb = F.normalize(user_emb)

        return torch.cat([user_emb, entity_emb], dim=0)

    def forward(self, cl=False, Drop=False):
        """
        通过整个模型进行前向传播。

        参数：
            - cl: 如果为True，返回用于对比学习的嵌入。
            - Drop: 如果为True，在前向传播过程中应用丢弃。

        返回：
            - 用户和物品的最终嵌入，可能包括用于对比学习的KG嵌入。
        """
        ego_embeddings = self.get_ego_embeddings()

        all_ig_embeddings = []
        all_kg_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ig_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings[:self.n_users + self.n_items])
            kg_embeddings = self.kg_forward(all_kg_embeddings[-1], Drop=Drop)

            ego_embeddings = kg_embeddings
            ego_embeddings[:self.n_items + self.n_users] += ig_embeddings
            all_ig_embeddings.append(ego_embeddings)
            all_kg_embeddings.append(kg_embeddings)

        # 从所有层聚合嵌入
        final_embeddings = torch.stack(all_ig_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        final_embeddings = final_embeddings[:self.n_users + self.n_items]
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])

        # 为对比学习聚合KG嵌入
        final_kg_embeddings = torch.stack(all_kg_embeddings, dim=1)
        final_kg_embeddings = torch.mean(final_kg_embeddings, dim=1)
        final_kg_embeddings = final_kg_embeddings[:self.n_users + self.n_items]
        user_kg_embeddings, item_kg_embeddings = torch.split(final_kg_embeddings, [self.n_users, self.n_items])

        if cl:
            return user_all_embeddings, item_all_embeddings, user_kg_embeddings, item_kg_embeddings
        return user_all_embeddings, item_all_embeddings

    def edge_sampling(self, h_index, t_index, r_index, rate=0.5):
        """
        从图中采样边以应用丢弃。

        参数：
            - h_index, t_index, r_index: 边的头、尾和关系索引。
            - rate: 采样率（保留此比例的边）。

        返回：
            - 采样后的头、尾和关系索引。
        """
        n_edges = h_index.shape[0]
        random_indices = np.random.choice(
            n_edges, size=int(n_edges * rate), replace=False
        )
        return h_index[random_indices], t_index[random_indices], r_index[random_indices]

    def calculate_loss(self, interaction):
        """
        计算包括主损失、对比损失和正则化损失在内的总损失。

        参数：
            - interaction: 用户-物品交互的批次。

        返回：
            - 主损失和SSL（自监督学习）损失的元组。
        """
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, \
            user_kg_emb, item_kg_emb = self.forward(cl=True, Drop=True)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # 正则化损失
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.entity_embedding(pos_item)
        neg_ego_embeddings = self.entity_embedding(neg_item)
        reg_loss = (torch.norm(u_ego_embeddings, p=2) + torch.norm(pos_ego_embeddings, p=2) \
                    + torch.norm(neg_ego_embeddings, p=2)) * self.reg_weight

        # 用于推荐的BPR损失
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)).sum()
        mf_loss = mf_loss + reg_loss

        # 对比学习损失
        ssl_loss2 = self.calculate_ssl_loss(user, pos_item, user_all_embeddings, user_kg_emb, item_all_embeddings,
                                            item_kg_emb)
        return mf_loss, ssl_loss2

    def calculate_ssl_loss(self, user, item, user_embeddings_v1,
                           user_embeddings_v2, item_embeddings_v1, item_embeddings_v2):
        """
        计算用于对比学习的自监督学习（SSL）损失。

        参数：
            - user, item: 此批次的用户和物品索引。
            - ..._v1, ..._v2: 用于对比比较的不同视图的嵌入。

        返回：
            - SSL损失。
        """
        norm_user_v1 = F.normalize(user_embeddings_v1[torch.unique(user)])
        norm_user_v2 = F.normalize(user_embeddings_v2[torch.unique(user)])
        norm_item_v1 = F.normalize(item_embeddings_v1[torch.unique(item)])
        norm_item_v2 = F.normalize(item_embeddings_v2[torch.unique(item)])

        # 用户的SSL损失
        user_pos_score = torch.mul(norm_user_v1, norm_user_v2).sum(dim=1)
        user_ttl_score = torch.matmul(norm_user_v1, norm_user_v2.t())
        user_pos_score = torch.exp(user_pos_score / self.tau)
        user_ttl_score = torch.exp(user_ttl_score / self.tau).sum(dim=1)
        user_ssl_loss = -torch.log(user_pos_score / user_ttl_score).sum()

        # 物品的SSL损失
        item_pos_score = torch.mul(norm_item_v1, norm_item_v2).sum(dim=1)
        item_ttl_score = torch.matmul(norm_item_v1, norm_item_v2.t())
        item_pos_score = torch.exp(item_pos_score / self.tau)
        item_ttl_score = torch.exp(item_ttl_score / self.tau).sum(dim=1)
        item_ssl_loss = -torch.log(item_pos_score / item_ttl_score).sum()

        ssl_loss = user_ssl_loss + item_ssl_loss

        return ssl_loss * self.cl_rate

    def predict(self, interaction):
        """
        预测给定用户-物品对的分数。

        参数：
            - interaction: 包含用户和物品ID。

        返回：
            - 预测的分数。
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        为给定用户预测所有物品的分数。

        参数：
            - interaction: 包含用户ID。

        返回：
            - 每位用户对所有物品的分数。
        """
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)