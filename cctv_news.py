import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

news_path = '/Users/xian.chen/data_bank/tradeA/combined_news_2010_to_2018_df.csv'
concept_detail_path = '/Users/xian.chen/data_bank/tradeA/combined_concept_detail.csv'
stock_path = '/Users/xian.chen/data_bank/tradeA/02price/price_20190331/'



def read_news_data(news_path, year_range):
    df = pd.read_csv(news_path, index_col=0)
    df.reset_index(inplace=True)
    df['date_index'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date_index',inplace=True)
    df = df[(df.index>year_range[0])&(df.index<str(int(year_range[1])+1))]
    df.dropna(inplace=True)
    return df

def read_news_detail_with_date(date_range):
    df = pd.read_csv(news_path, index_col=0)
    df.reset_index(inplace=True)
    df['date_index'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date_index',inplace=True)
    df = df[(df.index>=date_range[0])&(df.index<=str(int(date_range[1])))]
    df.dropna(inplace=True)
    return df

def read_news_with_keyword_count(date_range,dict_arg):
    df = read_news_detail_with_date(date_range)
    df[dict_arg['concept_word_e']] = df.content.str.contains(dict_arg['news_concept_word'])
    df = df[df[dict_arg['concept_word_e']] == True]
    pd.set_option('display.max_colwidth', -1)
    print('Top 3 date: ' + str(df.resample('D').count().sort_values(by=dict_arg['concept_word_e'], ascending=False).head(3).index.format()))
    print('The highest in on 2017-05-16, the keyword frequency is 26 times.')
    return #df.resample('D').count()

def read_news_with_keyword_count_and_details(date_range_for_reading,dict_arg):
    df = read_news_detail_with_date(date_range_for_reading)
    df[dict_arg['concept_word_e']] = df.content.str.contains(dict_arg['news_concept_word'])
    df = df[df[dict_arg['concept_word_e']] == True]
    pd.set_option('display.max_colwidth', -1)

    return df.iloc[:,1:]



def read_concept_detail_data(concept_detail_path, concept_word):
    concept_detail_df = pd.read_csv(concept_detail_path, index_col=0)
    concept_detail_df = concept_detail_df[concept_detail_df['concept_name']==concept_word]
    return concept_detail_df


def read_stock_price_data(stock_path, ts_stock_code):
    price_df = pd.read_csv(stock_path+ts_stock_code+'.csv', index_col=0)
    price_df['date'] = pd.to_datetime(price_df['trade_date'], format='%Y%m%d')
    price_df.set_index('date',inplace=True)
    price_df = price_df.iloc[::-1]
    price_df['daily_return'] = price_df.close.pct_change(1)
    price_df['weekly_return'] = price_df.close.pct_change(5)
    price_df['monthly_return'] = price_df.close.pct_change(20)
    price_df['two_month_return'] = price_df.close.pct_change(40)

    return price_df


def plotting(merged_df, concept_word_e, concept_index):
    plt.figure(figsize=(18, 5))
    top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    bottom = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    top.plot(merged_df.index, merged_df[['close']])
    # plt.plot(range(10))
    #     print(merged_df.head())
    top.grid(True)
    bottom.bar(merged_df.index, merged_df[concept_word_e], color='black')

    top.axvspan(datetime.strptime('20170516','%Y%m%d'), datetime.strptime('20170730','%Y%m%d'), color='green', alpha=0.2)
    bottom.axvspan(datetime.strptime('20170516','%Y%m%d'), datetime.strptime('20170520','%Y%m%d'), color='green', alpha=0.2)

    # top.axvspan(datetime.strptime('20180830','%Y%m%d'), datetime.strptime('20181030','%Y%m%d'), color='red', alpha=0.2)
    # bottom.axvspan(datetime.strptime('20180830','%Y%m%d'), datetime.strptime('20180910','%Y%m%d'), color='red', alpha=0.2)

    # set the labels
    top.axes.get_xaxis().set_visible(False)
    top.set_title(
        'Stock Price v.s. Keyword Frequency(' + 'Code: ' + str(concept_index[2]) + ' ' + '  Concept: ' + str(
            concept_word_e) + ')')
    top.set_ylabel('close price')
    bottom.set_ylabel('keyword_freq:')


def model_cctv_beta(news_path,
                    concept_word,
                    concept_word_e,
                    news_concept_word,
                    resample_freq,
                    #                     concept_path,
                    concept_detail_path,
                    #                     ts_stock_code,
                    stock_path,
                    year_range,
                    num_of_stocks
                    ):
    df = read_news_data(news_path, year_range)
    df[concept_word_e] = df.content.str.contains(news_concept_word)
    df = df[df[concept_word_e] == True]
    #     print(df.head())
    stat = df.resample(resample_freq).count()
    #     print(df)
    #     print(stat)
    #     print('ok1')
    ######################

    #     concept_df = read_concept_data(concept_path)
    #     concept_code = concept_df[concept_df.name==concept_word]['code'].values[0]
    #     print('ok2')
    ######################

    concept_detail_df = read_concept_detail_data(concept_detail_path, concept_word)
    concept_detail_df['Company English Name'] = None
    # concept_detail_df['Company English Name'][0:9] = np.array(['Shenzhen Yan Tian Port','Zoomlion','XCMG Construction Machinery Co., Ltd.','Zhuhai Port Holdings Group Co., Ltd', 'LiuGong, Shantui', 'SUFA Technology Industry Co., Ltd. CNNC', 'Xinjiang Tianshan Cement Co., Ltd.', 'China Dalian Intl Cooperat', 'Xiamen Port Development Co., Ltd.'])
    concept_detail_df.iloc[0:9,4] = np.array(['Shenzhen Yan Tian Port','Zoomlion','XCMG Construction Machinery Co., Ltd.','Zhuhai Port Holdings Group Co., Ltd', 'LiuGong, Shantui', 'SUFA Technology Industry Co., Ltd. CNNC', 'Xinjiang Tianshan Cement Co., Ltd.', 'China Dalian Intl Cooperat', 'Xiamen Port Development Co., Ltd.'])

    print(concept_detail_df[0:num_of_stocks].iloc[:,1:])
    for concept_index in concept_detail_df.values[0:num_of_stocks]:
        #     print(concept_row['ts_code'])
        #     print('ok3')
        ######################

        price_df = read_stock_price_data(stock_path, concept_index[2])
        #         print(price_df.head())
        sliced_df = price_df[(price_df.index > year_range[0]) & (price_df.index < str(int(year_range[1]) + 1))]
        resampled_df = sliced_df.resample(resample_freq).last()
        #     print('ok4')
        #     print(resampled_df)
        ######################

        merged_df = pd.merge(resampled_df, stat, left_index=True, right_index=True)
        merged_df.dropna(inplace=True)
        #     print('ok5')
        #     print(merged_df)
        ######################

        plotting(merged_df, concept_word_e, concept_index)

    #     print('ok6')
    return  # merged_df


def execute(dict_arg={
                      'concept_word' : '一带一路',
                      'concept_word_e' : 'one_belt_one_road',
                      'news_concept_word' : '一带一路',
                      'year_range' : ['2016','2018'],
                        'num_of_stocks': 5,
                        'resample_freq': 'D'

                    }):
    # print('Enter your concept word in English:')
    # dict_arg['concept_word_e'] = input()
    # print('Enter your concept word in Chinese:')
    # dict_arg['concept_word'] = input()
    # print('Enter the corresponding concept you want to look at in CCTV news:')
    # dict_arg['news_concept_word'] = input()
    # print('Enter year range you want to look at:')
    # dict_arg['year_range'] = input()



    concept_word = dict_arg['concept_word']
    concept_word_e = dict_arg['concept_word_e']
    news_concept_word = dict_arg['news_concept_word']
    year_range = dict_arg['year_range']

    num_of_stocks = dict_arg['num_of_stocks']
    resample_freq = dict_arg['resample_freq']

    model_cctv_beta(news_path,
                    concept_word,
                    concept_word_e,
                    news_concept_word,
                    resample_freq,
                    concept_detail_path,
                    #                     ts_stock_code,
                    stock_path,
                    year_range,
                    num_of_stocks)

    return #concept_word, concept_word_e, news_concept_word, year_range
