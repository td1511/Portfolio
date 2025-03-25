import wbgapi as wb
import pandas as pd

indicators ={
    #Inflation, GDP deflator (annual %)
    'Инфляция, дефлатор ВВП':'NY.GDP.DEFL.KD.ZG',
    #GDP
    'ВВП':'NY.GDP.MKTP.CN',
    #exports
    'Экспорт':'NE.EXP.GNFS.CN',
    #broad money
    'Денежная масса':'FM.LBL.BMNY.CN',
    #PPP conversion factor, GDP (LCU per international $)
    'Динамики ВВП по ППС':'PA.NUS.PPP',
    # Current account balance (BoP, current US$)
    'Баланс текущего счета':'BN.CAB.XOKA.CD',
    #NE.DAB.TOTL.CN	Gross national expenditure (current LCU)
    'Валовые национальные расходы':'NE.DAB.TOTL.CN'
}

list_indicator=[]
for i,j in zip(indicators.values(),indicators.keys()):
    if i =='NY.GDP.DEFL.KD.ZG' :
        df_indicator = wb.data.DataFrame(i,time = 2021, labels= True, skipAggs = True)
        list_indicator.append(df_indicator)
        continue 
    df_indicator = wb.data.DataFrame(i,time = range(2020,2022), labels= True, skipAggs = True) 
    d ={ 'Country' :df_indicator['Country'], 
                    j +',%' : ((df_indicator['YR2021']-df_indicator['YR2020'])/df_indicator['YR2020'])*100}
    a = pd.DataFrame(data = d)
    list_indicator.append(a)

#объединение фрейм
df = pd.concat(list_indicator,axis =1)
#Переименовать столбец
df = df.rename(columns = { 'NY.GDP.DEFL.KD.ZG':'Инфляция, дефлатор ВВП,%' })
#удалить совпадение столбцы но держать первую
df = df.loc[:,~df.T.duplicated(keep='first')]
#Преобразовать индекс в столбец
df.reset_index(inplace=True)
#установить столбец в индекс
df = df.set_index('Country')
#Сохранять в CSV
df.to_csv('data.csv')

df_1= df.dropna()

df_describe= df_1.describe().transpose()

df_describe.index.name ='Value'

df_describe.to_csv('describe.csv')

#Инфляция, дефлятор ВВП
df_deflator = wb.data.DataFrame('NY.GDP.DEFL.KD.ZG',time = range(2000,2022),skipAggs = True ,labels = True)

df_deflator = df_deflator.set_index('Country')

#инфляция потребительских це
df_cpi = wb.data.DataFrame('FP.CPI.TOTL.ZG',time = range(2000,2022),skipAggs = True ,labels = True)

df_cpi = df_cpi.set_index('Country')


df_deflator.to_csv('deflator.csv')

df_cpi.to_csv('cpi.csv')