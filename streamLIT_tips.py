import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main():
    st.set_page_config(page_title='My Streamlit TIPS dataset', layout='wide', page_icon="🧊")
    st.write(" ### Исследование по чаевым ")

    if st.button('Нажмите меня для + в вашу карму :)'):
        st.write('Да пребудет с вами сила!')
    st.write("""## Гистограмма по чекам""")
    path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
    df = pd.read_csv(path)
    st.dataframe(df)
    fig, ax = plt.subplots()
    ax.hist(df['total_bill'], color='skyblue', bins=10, edgecolor='white')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Count')
    ax.set_title('Гистограмма по чекам')
    st.pyplot(fig)

    st.write("""## График взимосвязи размера чека и чаевых в нем""")
    colors = np.arange(len(df))
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['total_bill'], df['tip'], c=colors, cmap='rainbow')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Tip')
    ax.set_title('Scatterplot of Total Bill vs Tip')
    st.pyplot(fig)

    st.write("""## График взимосвязи дня недели и размера чека""")
    fig, ax = plt.subplots()
    ax.scatter(df['total_bill'], df['day'], c=colors, cmap='cool')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Day of the week')
    ax.set_title('Scatterplot of Total Bill vs Day of the week')
    st.pyplot(fig)

    st.write("""
             ## Box plot c суммой всех счетов за каждый день, c разбивкой по time (Dinner/Lunch)
             """)

    grouped = df.groupby(['day', 'time']).sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([grouped[grouped['time'] == 'Lunch']['total_bill'], grouped[grouped['time'] == 'Dinner']['total_bill']],
           labels=['Lunch', 'Dinner'])
    ax.set_title('Box Plot of Total Bills by Time', fontsize=16)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Bill')
    st.pyplot(fig)

    st.write("""
             ## Гистограммы чаевых на обеде и ужине
             """)
    lunch_data = df[df['time'] == 'Lunch']['tip']
    dinner_data = df[df['time'] == 'Dinner']['tip']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(lunch_data, bins=25, color='blue', alpha=0.8, edgecolor='black')
    ax2.hist(dinner_data, bins=25, color='yellow', alpha=0.3, edgecolor='blue')
    ax1.set_xlabel('Tips')
    ax1.set_ylabel('Quantity of tips in Dollars')
    ax1.set_title('Lunch Tips')
    ax2.set_xlabel('Tips')
    ax2.set_ylabel('Quantity of tips in Dollars')
    ax2.set_title('Dinner Tips')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("""
             ## Взаимосвязь размера счета и чаевых, с дополнительной разбивкой по курящим/некурящим
             """)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    axes[0].scatter(df.loc[(df["sex"]=="Male") & (df["smoker"]=="No"), "total_bill"],
                df.loc[(df["sex"]=="Male") & (df["smoker"]=="No"), "tip"])
    axes[0].set_xlabel("Total Bill")
    axes[0].set_ylabel("Tip")
    axes[0].set_title("Мужчины, некурящие")
    axes[1].scatter(df.loc[(df["sex"]=="Female") & (df["smoker"]=="No"), "total_bill"],
                df.loc[(df["sex"]=="Female") & (df["smoker"]=="No"), "tip"])
    axes[1].set_xlabel("Total Bill")
    axes[1].set_ylabel("Tip")
    axes[1].set_title("Женщины, некурящие")
    axes[2].scatter(df.loc[(df["sex"]=="Male") & (df["smoker"]=="Yes"), "total_bill"],
                df.loc[(df["sex"]=="Male") & (df["smoker"]=="Yes"), "tip"])
    axes[2].set_xlabel("Total Bill")
    axes[2].set_ylabel("Tip")
    axes[2].set_title("Мужчины, курящие")
    axes[3].scatter(df.loc[(df["sex"]=="Female") & (df["smoker"]=="Yes"), "total_bill"],
                df.loc[(df["sex"]=="Female") & (df["smoker"]=="Yes"), "tip"])
    axes[3].set_xlabel("Total Bill")
    axes[3].set_ylabel("Tip")
    axes[3].set_title("Женщины, курящие")


    st.pyplot(fig)


if __name__ == '__main__':
    main()