#### 端口映射

#### 卷挂载

目录挂载

```bash
docker run -v /host/code:/app/code python  # 宿主机代码目录映射到容器[3,5](@ref)
```

单文件挂载

```bash
docker run -v /host/config.json:/app/config.json nginx  # 配置文件热更新[1](@ref)
```

权限处理

```bash
chmod 777 /host/code          # 解决容器内写权限问题
docker run -u root -v ...      # 以root用户运行容器[3](@re
```

**内存挂载（tmpfs）**：临时数据存储（如敏感信息处理）

```bash
docker run --tmpfs /tmp redis  # 数据仅存内存，容器退出后消失
```