import requests
from bs4 import BeautifulSoup


def get_movie_details(movie_name):
    # IMDb搜索URL
    search_url = "https://www.imdb.com/find?q="
    # 拼接完整的搜索URL
    search_url += movie_name.replace(' ', '+')

    # 发送请求获取搜索结果页面
    response = requests.get(search_url)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找第一个搜索结果的链接
    # first_result = soup.find('td', class_='result_text').find('a')
    # if first_result:
    #     movie_url = "https://www.imdb.com" + first_result['href']
    #
    #     # 发送请求获取电影详情页面
    #     movie_response = requests.get(movie_url)
    #     movie_soup = BeautifulSoup(movie_response.text, 'html.parser')
    #
    #     # 查找电影图片
    #     image_tag = movie_soup.find('div', class_='ipc-poster__poster-image').find('img')
    #     if image_tag:
    #         image_url = image_tag['src']
    #         # 获取图片
    #         image_response = requests.get(image_url)
    #         with open(f"{movie_name}.jpg", 'wb') as f:
    #             f.write(image_response.content)
    #         print(f"图片已保存为 {movie_name}.jpg")
    #     else:
    #         print("未找到电影图片")
    # else:
    #     print("未找到电影")


# 使用示例
get_movie_details("The Dark Knight")
