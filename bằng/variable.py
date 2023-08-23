import pandas as pd
import statsmodels.api as sm 
import plotly.express as px
from statsmodels.tools.eval_measures import rmse
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
df = pd.read_csv('data.csv')
df = df.dropna()
#sap xep ten cac nuoc theo chieu a-z
#df = df.sort_values("Country")

df_reg = df.drop(columns = ['economy','Country'])
#список имен независимых переменных
list_X=list(df_reg.columns[1:])

# зависимая переменная y (Инфляция, дефлатор ВВП):
Y = df_reg['Инфляция, дефлатор ВВП,%']
# ВВП
X1 = df_reg[[f'{list_X[0]}']]
# Экспорт
X2 =  df_reg[[f'{list_X[1]}']]
# денежная масса
X3 = df_reg[[f'{list_X[2]}']]
# Динамики по ППС, ВВП
X4 = df_reg[[f'{list_X[3]}']]
#Баланс текущего счета 
X5 = df_reg[[f'{list_X[4]}']]
#Валовые национальные расходы
X6 = df_reg[[f'{list_X[5]}']]
# множесвтенные показатели 
Z=df_reg[list_X]



'''fig_linear = px.scatter(df, x='ВВП,%', y="Инфляция, дефлатор ВВП,%", trendline="ols")
results = px.get_trendline_results(fig_linear)
print(results.px_fit_results.iloc[0].summary())'''

'''print(list_X)
X_multi = sm.add_constant(Z)
y_multi = y # y из variable
model_multi = sm.OLS(y_multi, X_multi).fit() 
results_multi= model_multi.summary()
print(results_multi)
ypred = model_multi.predict(X_multi)

rmse_multi = rmse(y_multi, ypred)

r2_multi = model_multi.rsquared
print(rmse_multi, r2_multi)
import numpy as np
from sklearn.metrics import mean_squared_error
X = df_reg.drop(columns = 'Инфляция, дефлатор ВВП,%', axis=1).values


X_train,X_test,y_train,y_test = train_test_split(Z.values,y.values,random_state=0)
model_lasso = Ridge(alpha = 0.001).fit(X_train,y_train)
ypred_lasso = model_lasso.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test,ypred_lasso))
r2_score_lasso = model_lasso.score(X_train,y_train)
equation_lasso = f'{model_lasso.intercept_:.2f} + {model_lasso.coef_[0]:.2f}*x1 + {model_lasso.coef_[1]:.2f} * x2 + {model_lasso.coef_[2]:.2f} * x3 + {model_lasso.coef_[3]:.2f} * x4 + {model_lasso.coef_[4]:.4f} * x5 + {model_lasso.coef_[5]:.2f} * x6'
text_lasso = 'R2 (R-квадрат) = ' + str(r2_score_lasso) + '\n \n' + 'Cреднеквадратическая ошибка RMSE = ' + str(rmse_lasso) + '\n \n' + 'Формула уравнения : y = '+ equation_lasso

print(rmse_lasso,equation_lasso,r2_score_lasso)'''