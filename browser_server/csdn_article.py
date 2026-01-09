import requests

# 原始获取文章列表的配置
cookies_get = {
    'uuid_tt_dd': '10_30868056540-1752190682383-488052',
    'fid': '20_16499080400-1752190683451-844434',
    'UserName': 'njsgcs',
    'UserInfo': '35bfae3693ed441fa1d055996074b62e',
    'UserToken': '35bfae3693ed441fa1d055996074b62e',
    'UserNick': 'njsgcs',
    'AU': '139',
    'UN': 'njsgcs',
    'BT': '1755303927588',
    'p_uid': 'U010000',
    'Hm_ct_6bcd52f51e9b3dce32bec4a3997715ac': '6525*1*10_30868056540-1752190682383-488052!5744*1*njsgcs',
    'FCNEC': '%5B%5B%22AKsRol_7Zpm3PD2GVjAUql8QwZ7KVGgqDC2XnU2t-1u5jz-N-NiHpZkVKyaSPOorqmYtKV88IOXtsZe5FHd7kXYDvhlg-BFXF3QOu9aHTmWSlfgI0wDXD82NxFkrmyc2N7Ubqiygd16ZEaApoT6D21yzPOucUJvfcA%3D%3D%22%5D%5D',
    'c_ins_fpage': '/index.html',
    'c_ins_um': '-',
    'ins_first_time': '1758008869778',
    'c_ins_prid': '1758008867819_531716',
    'c_ins_rid': '1758008882105_115535',
    'c_ins_fref': 'https://blog.csdn.net/weixin_43427721/article/details/134539003',
    'csdn_newcert_njsgcs': '1',
    'c_ab_test': '1',
    '__gads': 'ID=28b80783b4d378cd:T=1755521518:RT=1765444185:S=ALNI_MauS8hSYsPGNfTsmTfLpn_SJ3wAcw',
    '__gpi': 'UID=00001182474eeead:T=1755521518:RT=1765444185:S=ALNI_MazP1mgW_-qlmpEmtmP4ENB2DNq3Q',
    '__eoi': 'ID=efacf055736f2b30:T=1755521518:RT=1765444185:S=AA-AfjZSvhv1WwhSFJD8ZcElUmFP',
    'dc_sid': '6c8b5dcc32929f47484f65ceef683b9a',
    'dc_session_id': '11_1766071778677.173471',
    'c_first_ref': 'default',
    'c_first_page': 'https%3A//mp.csdn.net/',
    'c_segment': '4',
    'Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac': '1765275334,1765377128,1765444166,1766071782',
    'HMACCOUNT': '4121053C40F1833F',
    '_clck': '5c2q0w%5E2%5Eg1y%5E0%5E2056',
    'c_dsid': '11_1766072683979.380320',
    'creative_btn_mp': '3',
    'fe_request_id': '1766074048596_2336_4220381',
    'vip_auto_popup': '1',
    'c_pref': 'https%3A//misuseds.blog.csdn.net/%3Ftype%3Dblog',
    'c_ref': 'https%3A//so.csdn.net/so/search%3Fspm%3D1001.2014.3001.4498%26q%3Dcsdn%25E8%25AF%25BB%25E5%258F%2596%25E6%2596%2587%25E7%25AB%25A0%26t%3Dblog%26u%3D',
    'c_utm_term': 'csdn%E8%AF%BB%E5%8F%96%E6%96%87%E7%AB%A0',
    'referrer_search': '1766074176328',
    'c_utm_relevant_index': '10',
    'relevant_index': '10',
    'c_utm_medium': 'distribute.pc_search_result.none-task-blog-2%7Eblog%7Efirst_rank_ecpm_v1%7Erank_v31_ecpm-9-53189151-null-null.nonecase',
    'c_page_id': 'default',
    'log_Id_pv': '11',
    'log_Id_click': '15',
    'Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac': '1766074241',
    '_clsk': '2vvny4%5E1766074243457%5E6%5E0%5Ez.clarity.ms%2Fcollect',
    'log_Id_view': '367',
    'dc_tos': 't7h3u9',
}

headers_get = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'origin': 'https://mp.csdn.net',
    'priority': 'u=1, i',
    'referer': 'https://mp.csdn.net/',
    'sec-ch-ua': '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0',
    'x-ca-key': '203803574',
    'x-ca-nonce': '5139291b-05ff-46f2-a563-974ff2419287',
    'x-ca-signature': 'S2o4cFnUcx1V5ks4us9kbX5Us98S3sNVek4Ap8iTpZA=',
    'x-ca-signature-headers': 'x-ca-key,x-ca-nonce',
}

params = {
    'page': '1',
    'visible': 'vip',
    'status': 'all_v3',
    'pageSize': '20',
}

# 获取文章列表
response = requests.get(
    'https://bizapi.csdn.net/blog/phoenix/console/v1/article/list',
    params=params,
    cookies=cookies_get,
    headers=headers_get,
)

# 解析响应数据并提取 articleId
data = response.json()
article_list = data['data']['list']
article_ids = [article['articleId'] for article in article_list]

print("文章ID列表:")
print(article_ids)

# 设置文章可见性的配置
cookies_post = {
    'uuid_tt_dd': '10_30868056540-1752190682383-488052',
    'fid': '20_16499080400-1752190683451-844434',
    'UserName': 'njsgcs',
    'UserInfo': '35bfae3693ed441fa1d055996074b62e',
    'UserToken': '35bfae3693ed441fa1d055996074b62e',
    'UserNick': 'njsgcs',
    'AU': '139',
    'UN': 'njsgcs',
    'BT': '1755303927588',
    'p_uid': 'U010000',
    'Hm_ct_6bcd52f51e9b3dce32bec4a3997715ac': '6525*1*10_30868056540-1752190682383-488052!5744*1*njsgcs',
    'FCNEC': '%5B%5B%22AKsRol_7Zpm3PD2GVjAUql8QwZ7KVGgqDC2XnU2t-1u5jz-N-NiHpZkVKyaSPOorqmYtKV88IOXtsZe5FHd7kXYDvhlg-BFXF3QOu9aHTmWSlfgI0wDXD82NxFkrmyc2N7Ubqiygd16ZEaApoT6D21yzPOucUJvfcA%3D%3D%22%5D%5D',
    'c_ins_fpage': '/index.html',
    'c_ins_um': '-',
    'ins_first_time': '1758008869778',
    'c_ins_prid': '1758008867819_531716',
    'c_ins_rid': '1758008882105_115535',
    'c_ins_fref': 'https://blog.csdn.net/weixin_43427721/article/details/134539003',
    'csdn_newcert_njsgcs': '1',
    'c_ab_test': '1',
    '__gads': 'ID=28b80783b4d378cd:T=1755521518:RT=1765444185:S=ALNI_MauS8hSYsPGNfTsmTfLpn_SJ3wAcw',
    '__gpi': 'UID=00001182474eeead:T=1755521518:RT=1765444185:S=ALNI_MazP1mgW_-qlmpEmtmP4ENB2DNq3Q',
    '__eoi': 'ID=efacf055736f2b30:T=1755521518:RT=1765444185:S=AA-AfjZSvhv1WwhSFJD8ZcElUmFP',
    'dc_sid': '6c8b5dcc32929f47484f65ceef683b9a',
    'dc_session_id': '11_1766071778677.173471',
    'c_first_ref': 'default',
    'c_first_page': 'https%3A//mp.csdn.net/',
    'c_segment': '4',
    'Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac': '1765275334,1765377128,1765444166,1766071782',
    'HMACCOUNT': '4121053C40F1833F',
    '_clck': '5c2q0w%5E2%5Eg1y%5E0%5E2056',
    'c_dsid': '11_1766072683979.380320',
    'creative_btn_mp': '3',
    'fe_request_id': '1766074048596_2336_4220381',
    'vip_auto_popup': '1',
    'c_utm_term': 'csdn%E8%AF%BB%E5%8F%96%E6%96%87%E7%AB%A0',
    'referrer_search': '1766074176328',
    'c_utm_relevant_index': '10',
    'relevant_index': '10',
    'c_utm_medium': 'distribute.pc_search_result.none-task-blog-2%7Eblog%7Efirst_rank_ecpm_v1%7Erank_v31_ecpm-9-53189151-null-null.nonecase',
    '_clsk': '2vvny4%5E1766074880867%5E7%5E0%5Ez.clarity.ms%2Fcollect',
    'c_page_id': 'default',
    'c_pref': 'https%3A//mp.csdn.net/',
    'c_ref': 'https%3A//misuseds.blog.csdn.net/',
    'log_Id_pv': '15',
    'Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac': '1766075268',
    'log_Id_view': '418',
    'log_Id_click': '19',
    'dc_tos': 't7h4f1',
}

headers_post = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'content-type': 'application/json;',
    'origin': 'https://mp.csdn.net',
    'priority': 'u=1, i',
    'referer': 'https://mp.csdn.net/',
    'sec-ch-ua': '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0',
    'x-ca-key': '203803574',
    'x-ca-nonce': '9c093f0e-3094-4fbb-92f6-de487de2ef3f',
    'x-ca-signature': 'lTCAx8UooUMP5Hggj/1eVZCBRNoUXtJbZrSm0wnMSUc=',
    'x-ca-signature-headers': 'x-ca-key,x-ca-nonce',
}

# 对每个 articleId 执行 POST 请求
for article_id in article_ids:
    json_data = {
        'articleId': article_id,
        'visible': 'all',
    }
    
    response = requests.post(
        'https://bizapi.csdn.net/blog/phoenix/console/v2/article/set-visible-range',
        cookies=cookies_post,
        headers=headers_post,
        json=json_data,
    )
    
    print(f"文章 {article_id} 设置可见性结果: {response.status_code}")
    if response.status_code == 200:
        print(f"成功设置文章 {article_id} 的可见性")
    else:
        print(f"设置文章 {article_id} 可见性失败: {response.text}")