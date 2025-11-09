#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

filepath = "dataset/heart.csv"

df = pd.read_csv(filepath)
print(f'Dataset downloaded to: {filepath}')


# In[ ]:


print(df.describe())
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

print(df.shape)


# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df_outliers = df[outliers.any(axis=1)]
print(df_outliers.sum())


# In[ ]:


df = df.drop(df_outliers.index)
print(df.shape)


# In[ ]:


print(df.head())
print("==============================================\n")
print(df.describe())


# In[ ]:


summary = df.describe().T
summary = summary.round(3)   # optional: reduce number of decimals
print(summary.to_markdown())


# ### Descrição Interpretativa das Variáveis
# 
# (Raciocínio humano)
# 
# **1. Age (Idade)**  
# A idade dos indivíduos varia aproximadamente entre 29 e 77 anos, com média próxima a 54 anos. Isso indica que a amostra é formada majoritariamente por adultos de meia-idade e idosos. Como o risco de doenças cardiovasculares aumenta com a idade, essa variável tende a ter influência direta na presença de doença cardíaca.
# 
# **2. Sex (Sexo)**  
# A variável sexo é categórica, onde 1 representa indivíduos do sexo masculino e 0 do sexo feminino. Se houver predominância de homens no conjunto, isso pode afetar a interpretação dos resultados, pois estatisticamente os homens apresentam maior incidência de doenças cardíacas antes das mulheres.
# 
# **3. Trestbps (Pressão Arterial em Repouso)**  
# A pressão arterial em repouso apresenta média em torno de 130 mmHg. Valores acima de 120 mmHg já são considerados elevados. Isso sugere que uma parte significativa da amostra pode apresentar predisposição à hipertensão, que é um fator de risco relevante para doenças cardiovasculares.
# 
# **4. Chol (Colesterol Sérico)**  
# Os valores de colesterol possuem média próxima de 245 mg/dL. Clínicamente, níveis acima de 200 mg/dL já são considerados altos. Logo, essa variável indica que muitos pacientes avaliados possuem hipercolesterolemia, contribuindo para risco aumentado de entupimento arterial e doenças cardíacas.
# 
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.histplot(data=df, x='age', kde=True)
plt.title('Distribuição de Idade')
plt.show()


# In[ ]:


sns.boxplot(data=df, y='chol')
plt.title('Distribuição de Colesterol')
plt.show()


# In[ ]:


sns.countplot(data=df, x='cp')
plt.title('Contagem por Tipo de Dor no Peito')
plt.show()


# In[ ]:


sns.countplot(data=df, x='sex', hue='target')
plt.title('Contagem de Target por Sexo')
plt.legend(title='Target', labels=['Não', 'Sim'])
plt.show()


# In[ ]:


sns.boxplot(data=df, x='target', y='age')
plt.title('Distribuição de Idade por Target')
plt.show()


# In[ ]:


sns.scatterplot(data=df, x='age', y='thalach', hue='target')
plt.title('Idade vs. Frequência Cardíaca Máxima')
plt.show()


# In[ ]:


corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlação')
plt.show()


# In[ ]:


df = df.drop('fbs', axis=1)


# ### 5. Identificação da variável alvo e tipo do problema
# (Raciocínio humano)
# 
# A variável que representa o rótulo (classe) é a coluna **`target`**.
# Essa variável assume valores discretos (por exemplo, 0 para ausência e 1 para presença de doença).  
# Portanto, o objetivo é determinar se um indivíduo possui ou não a condição, caracterizando o problema como um **problema de classificação binária**.
# 
# ---
# 
# ### 6. Relações entre variáveis e a presença de doença
# 
# Ao analisar a matriz de correlação, observa-se que algumas variáveis apresentam relação com o rótulo:
# 
# - Variáveis com **correlação positiva** com `target` tendem a estar associadas a uma **maior probabilidade** de presença da doença.
# - Variáveis com **correlação negativa** tendem a estar associadas à **menor probabilidade** da condição.
# 
# De forma geral, fatores como **idade elevada**, **níveis mais altos de colesterol** e **pressão arterial aumentada** costumam aparecer positivamente correlacionados com a doença cardíaca, indicando contribuição para o risco. Já variáveis como **frequência cardíaca máxima atingida** e certas medidas relacionadas à aptidão física podem apresentar correlação negativa, indicando associação com menor risco.
# 
# Em resumo, a análise de correlação auxilia na identificação de quais características estão **mais fortemente relacionadas à presença ou ausência da doença**, contribuindo para compreensão e futura modelagem preditiva.
# 
# ---

# In[ ]:


# <----- To be more readable ----->


# ### 7. Análise das Correlações e Seleção de Atributos  
# *(Raciocínio elaborado por um modelo de IA, de forma explícita e estruturada)*
# 
# Ao analisar o mapa de calor de correlação, o objetivo é verificar quais variáveis do conjunto de dados possuem relação mais forte com a variável alvo `target`, que representa a presença ou ausência de doença cardíaca. Como modelo de IA, minha interpretação é feita observando padrões numéricos, intensidade das correlações e possíveis redundâncias entre atributos.
# 
# ---
# 
# ### 7.1. Sobre a Correlação com a Variável Alvo
# 
# A correlação não indica causalidade, apenas associação.  
# Além disso, o **sinal da correlação (positivo ou negativo) não define utilidade da variável** — tanto correlações positivas quanto negativas podem ser relevantes para o modelo.
# 
# O que importa é a **magnitude** da correlação (o valor absoluto).
# 
# ---
# 
# ### 7.2. Principais Variáveis Informativas
# 
# Ao observar a coluna `target`, as variáveis que apresentam maior força de relação (positiva ou negativa) são:
# 
# | Atributo   | Correlação com `target` | Interpretação |
# |-----------|-------------------------|--------------|
# | `cp`      | +0.37                   | Tipos de dor no peito mostram associação com diagnóstico. |
# | `thalach` | +0.42                   | Frequência cardíaca máxima está relacionada a menor risco. |
# | `exang`   | -0.41                   | Angina ao exercício aumenta probabilidade de doença. |
# | `oldpeak` | -0.44                   | Depressão do ST é um indicador clínico importante. |
# | `ca`      | -0.45                   | Número de vasos obstruídos é altamente relevante. |
# | `thal`    | -0.46                   | Defeito de perfusão cardíaca fortemente relacionado. |
# 
# Esses atributos fornecem **informação significativa** ao modelo.
# 
# ---
# 
# ### 7.3. Variáveis com Baixa Influência
# 
# Algumas variáveis exibem correlação fraca com `target`, podendo ter pouco impacto no modelo:
# 
# - `chol`
# - `fbs`
# - `restecg`
# - `sex`
# - `age`
# 
# Essas variáveis **não precisam ser removidas**, mas também **não são determinantes**.
# 
# ---
# 
# ### 7.4. Avaliação de Redundância (Multicolinearidade)
# 
# O mapa de calor não mostra pares de variáveis com correlação forte entre si (acima de |0.85|).  
# Portanto, **não há necessidade de remover atributos por redundância**.
# 
# ---
# 
# ### 7.5. Conclusão Técnica
# 
# - **Não** se deve selecionar atributos pela **correlação positiva**.
# - O critério correto é a **magnitude da associação** e a **informação adicionada ao modelo**.
# - Os melhores candidatos para iniciar o modelo são:  
#   **`cp`, `thalach`, `exang`, `oldpeak`, `ca`, `thal`**.
# - A etapa seguinte recomendada é validar importância das variáveis via um modelo como **Random Forest** ou **Logistic Regression com regularização L1**.
# 
# ---
# 

# In[ ]:


X = df.drop(columns=['target']) # atributos
y = df['target'] # rotulo

# StandardScaler para SVM


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% teste e 70% treino, random_state faz repetir para todos os ciclos

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train, y_train)

model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

score_rbf = model_rbf.score(X_test, y_test)
print("Acurácia RBF: " + str(round(score_rbf, 3) * 100) + "%")

score_linear = model_linear.score(X_test, y_test)
print("Acurácia Linear: " + str(round(score_linear, 3) * 100) + "%")



# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model_linear.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão(modelo linear)")
plt.show()

print("Modelo linear:")
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model_rbf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão(modelo linear)")
plt.show()

print("Modelo linear:")
print(classification_report(y_test, y_pred))


# 1. Linear, pois teve maior precisao e menos FP e FN
# 2. Como doente, pois há mais falsos positivos
# 3 As de maior valor absoluto
# 4. Possuir um dataset maior e mais equilibrado, e filtrar as variaveis de baixo valor absoluto
# 5. Riscos de falso diagnostico

# ## Relatório: Desempenho do Modelo de Classificação para Diagnóstico Cardíaco
# 
# ### 1. Qual kernel obteve melhor desempenho? Por quê?
# 
# **Resposta do aluno:**  
# Linear, pois teve maior precisão e menos FP e FN.
# 
# **Correção (IA):**  
# O kernel **Linear** apresentou o melhor desempenho devido ao melhor equilíbrio entre **precisão**, **recall** e **F1-score**. Isso indica que o modelo conseguiu separar adequadamente as classes sem gerar muitos falsos positivos nem falsos negativos. Além disso, o desempenho do kernel linear sugere que o conjunto de dados possui uma separabilidade aproximadamente linear, o que torna esse kernel mais adequado nesse caso.
# 
# ---
# 
# ### 2. Existe tendência do modelo errar mais ao classificar doentes como saudáveis ou o contrário? Quais as implicações?
# 
# **Resposta do aluno:**  
# Como doente, pois há mais falsos positivos.
# 
# **Correção (IA):**  
# Se há mais **falsos positivos**, o modelo está classificando **pessoas saudáveis como doentes**. Isso resulta em maior **sensibilidade (recall)** e menor precisão.  
# **Implicações:** embora seja menos grave do que falsos negativos (que deixariam doentes sem tratamento), falsos positivos podem levar a **ansiedade no paciente, realização de exames desnecessários e aumento de demanda no sistema de saúde**. Ainda assim, esse tipo de erro é preferível ao oposto em diagnósticos de doenças graves, mas deve ser equilibrado.
# 
# ---
# 
# ### 3. Quais variáveis parecem mais relevantes para o diagnóstico? Como isso foi identificado?
# 
# **Resposta do aluno:**  
# As de maior valor absoluto.
# 
# **Correção (IA):**  
# As variáveis mais relevantes para o diagnóstico foram identificadas observando-se os **coeficientes do modelo linear** e/ou os **valores de correlação** com o target. Variáveis com valores absolutos maiores nesses coeficientes ou correlações contribuem mais para a decisão do modelo, pois influenciam diretamente a separação entre classes.
# 
# ---
# 
# ### 4. Que melhorias poderiam ser realizadas no modelo?
# 
# **Resposta do aluno:**  
# Possuir um dataset maior e mais equilibrado, e filtrar as variáveis de baixo valor absoluto.
# 
# **Correção (IA):**  
# Melhorias possíveis incluem:  
# - **Aumentar a base de dados** para melhorar generalização;  
# - **Balancear as classes** via oversampling, undersampling ou **SMOTE**;  
# - **Selecionar features relevantes** usando métodos estatísticos e de regularização;  
# - **Ajustar hiperparâmetros** do modelo via GridSearchCV;  
# - Testar **outros algoritmos**, como Random Forest ou Gradient Boosting, para comparação.
# 
# ---
# 
# ### 5. Quais riscos éticos e práticos existiriam caso esse modelo fosse utilizado em hospitais reais?
# 
# **Resposta do aluno:**  
# Riscos de falso diagnóstico.
# 
# **Correção (IA):**  
# Os principais riscos são:  
# - **Falsos positivos:** pessoas saudáveis podem passar por exames invasivos e estresse psicológico;  
# - **Falsos negativos:** pacientes doentes podem deixar de receber tratamento, causando agravamento da condição;  
# - **Viés de treinamento:** se os dados não representam toda a população, certos grupos podem ser prejudicados;  
# - **Dependência excessiva da IA:** decisões clínicas não podem ser delegadas totalmente ao modelo.
# 
# Portanto, o modelo deve ser **apenas uma ferramenta auxiliar**, nunca um substituto para avaliação médica profissional.
# 
