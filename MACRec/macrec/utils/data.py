import json
import torch
import numpy as np
import pandas as pd


def append_his_info(dfs: list[pd.DataFrame], summary: bool = False, neg: bool = False) -> list[pd.DataFrame]:
    """주어진 데이터프레임들에 사용자의 이력 정보를 추가합니다.
        히스토리 정보 추가를 위해 사용자 아이디, 아이템 아이디, 평점, 타임스탬프 컬럼이 필요합니다.
        이력 정보는 사용자가 평가한 아이템들의 리스트와 평점들의 리스트로 구성됩니다.
        요약 정보를 추가할 경우, 사용자가 평가한 아이템들의 요약 정보를 추가합니다.

    Args:
        dfs (list[pd.DataFrame]): 데이터프레임들의 리스트입니다.
        summary (bool, optional): 요약 정보를 추가할지 여부를 나타내는 불리언 값입니다. 기본값은 False입니다.
        neg (bool, optional): 부정적인 항목을 제외할지 여부를 나타내는 불리언 값입니다. 기본값은 False입니다.

    Returns:
        list[pd.DataFrame]: 이력 정보가 추가된 데이터프레임들의 리스트입니다.
    """
    all_df = pd.concat(dfs)
    # sort_df 데이터프레임에 추가하게 된다. 
    sort_df = all_df.sort_values(by=["timestamp", "user_id"], kind="mergesort") # timestamp가 동일한 경우에만, user_id가 정렬 기준으로 사용됨 
    position = []
    user_his = {}
    history_item_id = []
    user_his_rating = {}
    history_rating = []
    for uid, iid, r, t in zip(sort_df["user_id"], sort_df["item_id"], sort_df["rating"], sort_df["timestamp"]):
        if uid not in user_his:
            # 초기화
            user_his[uid] = []
            user_his_rating[uid] = []
        position.append(len(user_his[uid])) # 현재까지의 이력 길이
        history_item_id.append(user_his[uid].copy()) # 현재까지의 이력 아이템
        history_rating.append(user_his_rating[uid].copy()) # 현재까지의 이력 평점
        user_his[uid].append(iid)
        user_his_rating[uid].append(r)
    sort_df["position"] = position
    sort_df["history_item_id"] = history_item_id
    sort_df["history_rating"] = history_rating
    if summary:
        user_his_summary = {}
        history_summary = []
        for uid, s in zip(sort_df["user_id"], sort_df["summary"]):
            if uid not in user_his_summary:
                # 초기화
                user_his_summary[uid] = []
            history_summary.append(user_his_summary[uid].copy())
            user_his_summary[uid].append(s)            
        sort_df["history_summary"] = history_summary
    ret_dfs = []
    for df in dfs:
        if neg:
            df = df.drop(columns=["neg_item_id"])
        if summary:
            df = df.drop(columns=["summary"])
        df = pd.merge(left=df, right=sort_df, on=["user_id", "item_id", "rating", "timestamp"], how="left")
        ret_dfs.append(df)
    del sort_df
    return ret_dfs