import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import plotly.express as px
import shap

st.title("Preditor de preço para planos de saúde")


st.write('#### Carregamento dos dados') 
st.write('Vamos carregar o data set e dar uma olhada nas colunas e qual tipo de dados elas carregam')
data = pd.read_csv('insurance.csv')

#Mudar valores de algumas colunas em inglês
data['sex'] = data['sex'].replace(['male', 'female'], ['masculino', 'feminino'])
data['region'] = data['region'].replace(['southeast', 'southwest', 'northeast', 'northwest'], ['sudeste', 'sudoeste', 'noroeste', 'nordeste'])
data['smoker'] = data['smoker'].replace(['yes', 'no'], ['sim', 'nao'])
data.rename(columns={'age':'idade', 'sex':'genero', 'bmi':'imc', 'children':'filhos', 'smoker':'fumante', 'region':'regiao', 'charges':'total'}, inplace=True)

st.write(data.head(15))

# Sidebar
# Especifique os parametros de entrada
st.sidebar.write('---')
st.sidebar.header('Especifique os parâmetros de entrada para Regressão linear')

def user_input_features():
    imc = st.sidebar.slider('Bmi',float(data.imc.min()), float(data.imc.max()), float(data.imc.mean()))
    idade = st.sidebar.slider('Idade', int(data.idade.min()), int(data.idade.max()), int(data.idade.mode()))
    filhos = st.sidebar.slider('Filhos', int(data.filhos.min()), int(data.filhos.max()), int(data.filhos.mode()))
    genero = st.sidebar.selectbox('Gênero', ('feminino', 'masculino'))
    fumante = st.sidebar.selectbox('Fumante', ('sim', 'nao'))
    regiao = st.sidebar.selectbox('Região', ('sudoeste', 'sudeste', 'noroeste', 'nordeste'))

    df = {'imc': imc,
          'idade': idade,
          'filhos': filhos,
          'genero': genero,
          'fumante': fumante,
          'regiao' : regiao
            }
    features = pd.DataFrame(df, index=[0])
    return features

df = user_input_features()

# Mostra os parametros escolhidos 
st.subheader('Parametros selecionados')
st.write(df)
st.write('---')

# cria as variáveis X e y
X = data.drop('total', axis=1)
y = data.total

# convert as variaveis categoricas 
new_df = pd.concat([X, df], ignore_index=True) #concatenar os dados para converter as variáveis categoricas
new_df = pd.get_dummies(new_df)
X = new_df.drop([1338])
df = new_df.loc[[1338]]


# define o modelo que será urilizado na regressão
model = XGBRegressor().fit(X, y)
pred = model.predict(df)

st.subheader('Predições')
st.markdown('Custo do plano de acordo com os valores escolhidos')
st.write(pred)
st.write('---')


st.set_option('deprecation.showPyplotGlobalUse', False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if st.button('Show SHAP Graphs'):
    st.header('Importância de cada feature')
    plt.title('Importância das features baseadas nos valores SHAP')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')
    plt.title('Importância das features baseadas nos valores SHAP')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')

st.subheader('Conclusão')
st.write('''De acordo com os valores Shap, a variável que mais influencia na composição do valor
            é se a pessoa é fumante ou não. Idade também colabora bastante no preço já que nos planos
            de saúde quanto mais velho uma pessoa é, mais caro é o valor''') 
