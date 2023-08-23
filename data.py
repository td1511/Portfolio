import tkinter as tk
from tkinter import ttk, Frame, Menu, Scrollbar,Canvas, messagebox
from pandastable import Table
import pandas as pd
import plotly.express as px
from PIL import Image, ImageTk
import kaleido
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import statsmodels.api as sm 
from variable import X1,X2,X3,X4,X5,X6,Y,Z
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error


#from matplotlib.figure import Figure
class ScrollFrame :
	def __init__(self, window, bg= "white",  font= ("Arial",10), fg = 'white', expand= 0, width= 0) :
		# Main Parameters For Scrollable Frame, Основные параметры прокручиваемой рамки
		self.root = root
		self.root.state('zoomed')
		self.BG , self.FG = bg , fg
		self.FONT = font
		self.WIDTH , self.EXPAND = width , expand
		# Main Frame Основная рама
		self.main_frame = Frame(self.root, bg= self.BG, borderwidth= 0)
		self.main_frame.pack(fill= 'both', expand= self.EXPAND)
		# Main Canvas, ScrollBar & Frame
		self.scrollbar = Scrollbar(self.main_frame, orient= 'vertical')
		self.canvas = Canvas(self.main_frame, bg= self.BG, borderwidth= 0)
		self.second_frame = Frame(self.canvas, bg= self.BG, borderwidth= 0)
		# Setting Up The Canvas & Frame
		self.scrollbar.configure(command= self.canvas.yview)
		self.canvas.configure(yscrollcommand= self.scrollbar.set)
		# Binding For Animations Of MouseWheel Привязка для анимации MouseWheel
		self.canvas.bind("<Enter>", self.__bound_mousewheel)
		self.canvas.bind("<Leave>", self.__unbound_mousewheel)
		screen_width = self.root.winfo_screenwidth()
		# Binding For Click Move Привязка для клика движение
		self.second_frame.bind("<Configure>" ,
					    lambda e : self.canvas.configure(scrollregion= self.canvas.bbox('all')))
		self.canvas.create_window((0,0), window= self.second_frame, width = screen_width - 18)
		# Paking Widgets On Screen Упаковка виджетов на экран
		self.scrollbar.pack(side= 'right', fill= 'y')
		self.canvas.pack(fill= 'both', expand= 1)
	def __bound_mousewheel(self, eve) :
		self.canvas.bind_all("<MouseWheel>", self.__move)
	def __unbound_mousewheel(self, eve) :
		self.canvas.unbind_all("<MouseWheel>")
	def __move(self, event) :
		self.canvas.yview_scroll(int(-1 *(event.delta / 120)), "units")

font_title  = ('Arial',20)
# For Demostration Pusposes
if __name__ == '__main__':
    root = tk.Tk()
    f = ScrollFrame(root, expand= True)

    screen_width = root.winfo_screenwidth()
    
    screen_height = root.winfo_screenheight()
   
    root.title('Моделирование и анализ динамики инфляционных процессов на странах мира')

    #Читать файл CSV
    df_full = pd.read_csv('data.csv')
   #удалить строки , содержещые пропущенные значение
    df = df_full.dropna()
     #sap xep ten cac nuoc theo chieu a-z
    df_1 = df
    df = df.sort_values("Country")
    df1 = df
    
    df_describe= pd.read_csv('describe.csv')

    
    df_reg = df.drop(columns = ['economy','Country'])

    df_full = df_full.drop(columns = ['economy','Country'])
    
    df_corr = df_full.corr(method='pearson')

    df_deflator = pd.read_csv('deflator.csv')

    df_deflator = df_deflator.set_index('Country')

    df_cpi = pd.read_csv('cpi.csv')

    df_cpi = df_cpi.set_index('Country')

    # создать frame , содерживающий data
    label_data = tk.Label(f.second_frame, text = 'Таблица данных',font = font_title , bg = 'white')
    label_data.pack(fill = 'both',expand= True)

    frame_data = tk.Frame(f.second_frame,bg = 'white')
    frame_data.pack(fill = 'both',expand= True)
    pt = Table(frame_data, dataframe=df1, showtoolbar=True, showstatusbar=True)
    pt.setRowHeight(40)
    pt.show()


    label_describe = tk.Label(f.second_frame, text = 'Создание описательной статистики',font = font_title , bg = 'white')
    label_describe.pack(fill = 'both',expand= True)

    frame_describe = tk.Frame(f.second_frame,bg ='white')
    frame_describe.pack(pady = 20)

    pt1 = Table(frame_describe, dataframe=df_describe, width = 1000)
    pt1.setRowHeight(30)
    pt1.show()

    
    #Коэффициент корреляции Пирсона
    label_pearson = tk.Label(f.second_frame, text = 'Коэффициент корреляции Пирсона', font = font_title,bg='white')

    label_pearson.pack(pady =(30,10),fill = 'both',expand= True)

    df_corr = df_full.corr(method='pearson')

    fig_corr = px.imshow(df_corr, text_auto='.3f',color_continuous_scale='rdbu', width=800)

    fig_corr.update_layout( margin=dict(l=20, r=20, t=10, b=0))

    fig_corr.write_image("fig_corr.png", engine="kaleido")

    test_corr = ImageTk.PhotoImage(Image.open("fig_corr.png"))

    label_corr = tk.Label(f.second_frame,image=test_corr, bg ='white')

    label_corr.image = test_corr

    label_corr.pack(fill= 'both',expand=True)
            
    #простая линейные регрессия

    frame_linear = tk.Frame(f.second_frame, bg ='white')
    frame_linear.pack(fill= 'both',expand = True)

    label_title_linear = tk.Label(frame_linear, text = 'Простая линейные регрессия - Simple Linear Regression', font = font_title, bg ='white')
    label_title_linear.pack(pady = (30,20), fill= 'both',expand = True)


    frame_combobox = tk.Frame(frame_linear,bg ='white')
    frame_combobox.pack(fill= 'both',expand = True)

    label_combobox = tk.Label(frame_combobox, bg = 'white',text = 'Выбор показатель ', font = ('Arial',12))
    label_combobox.grid(padx = (screen_width/2-200,0),column=0, row = 0, sticky='e')


    indicator = list(df.columns[3:])
    indicator.insert(0," ")
    combobox = ttk.Combobox(frame_combobox,values=indicator, width=50)
    combobox.current(0) # установить значение по умолчанию пустой


    combobox.grid(column=1, row = 0, sticky='w')
    frame_linear_1 = tk.Frame(frame_linear, bg ='white')
    frame_linear_1.pack(fill= 'both',expand = True)
    label_linear = tk.Label(frame_linear_1)
    label_summary = tk.Label(frame_linear_1)
    label_residuals = tk.Label(frame_linear_1)
    
    #RMSE простой линейная регрессия
    from statsmodels.tools.eval_measures import rmse



    columns_name_simple = list(df.columns[3:])
    columns_name_simple.insert(0,'Показатель')
    
    df_result_simple = pd.DataFrame(columns = columns_name_simple)

    df_result_simple['Показатель'] = ['R2 (R-квадрат)','Cреднеквадратическая ошибка RMSE','Формула уравнения']


    def selected(event):
        try:
            global label_linear,label_summary, frame_linear_1, label_residuals
            exists_1 = frame_linear_1.winfo_exists() 
            if exists_1 == 0:
                frame_linear_1= tk.Frame(frame_linear, bg ='white')
                frame_linear_1.pack(fill='both', expand =True) 
            label_linear.destroy()
            label_summary.destroy()
            label_residuals.destroy()
            # получаем выделенный элемент
            selection = combobox.get()
            
            fig_linear = px.scatter(df_full, x=selection, y="Инфляция, дефлатор ВВП,%", trendline="ols")
            fig_linear.update_layout(
                    margin=dict(l=0, r=0, t=50, b=0),
                    legend=dict(orientation="h",
                                yanchor="bottom",
                                y=0.99,
                                xanchor="right",
                                x=1))
            #How to embed data like regression results into legend?
            model = px.get_trendline_results(fig_linear)
            r2_rsquared_simple = model.px_fit_results.iloc[0].rsquared
            alpha = model.iloc[0]["px_fit_results"].params[0]
            beta = model.iloc[0]["px_fit_results"].params[1]
            fig_linear.data[1].line.color = 'red'
            fig_linear.data[0].name = 'Наблюдение'
            fig_linear.data[0].showlegend = True
            fig_linear.data[1].name = '\n'+'Инфляция, дефлатор ВВП = ' + str(round(alpha, 3)) + ' + ' + str(round(beta, 3)) + "*"+ selection
            fig_linear.data[1].showlegend = True
            fig_linear.write_image("fig_linear.png", engine="kaleido")


            
            img = Image.open("fig_linear.png")
            test_linear = ImageTk.PhotoImage(img)
            label_linear = tk.Label(frame_linear_1,image=test_linear, bg ='white')
            label_linear.image = test_linear
            label_linear.pack(fill= 'both',expand=True)
            
            
            #How to display statsmodels model summary in tkinter?
            
            
            results = px.get_trendline_results(fig_linear)
            plt.rc('figure', figsize=(10, 4.1))
            plt.figure().set_figwidth(8)
            
            plt.text(0, 1, str(results.px_fit_results.iloc[0].summary(xname = ['Коэффициент',selection], 
                                                                        yname= 'Инфляция, дефлатор ВВП,%')), 
                                                                        {'fontsize': 9.5}, fontproperties = 'monospace')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('fig_summary.png')
            test_summary = ImageTk.PhotoImage(Image.open("fig_summary.png"))
            label_summary = tk.Label(frame_linear_1,image=test_summary, bg ='white')
            label_summary.image = test_summary
            label_summary.pack(fill= 'both',expand=True)
            

            # Удалить данных старший figure, чтобы созадать новый
            plt.figure(clear=True)
            #Plot the residuals of a linear regression. Построить остатки линейной регрессии
            sns.residplot(data=df, x=selection, y="Инфляция, дефлатор ВВП,%").set(title='График остатка - Residuals')
            plt.savefig('fig_residuals.png')
            test_residuals = ImageTk.PhotoImage(Image.open("fig_residuals.png"))
            label_residuals = tk.Label(frame_linear_1,image=test_residuals, bg ='white')
            label_residuals.image = test_residuals
            label_residuals.pack(fill= 'both',expand=True)



            X = df_1[selection]
            y = Y

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            ypred = model.predict(X)

            # calc rmse
            rmse_simple = rmse(y, ypred)


            name = df.columns[3:]
            for i in name:
                if selection == i:
                    df_result_simple.loc[0,i] = r2_rsquared_simple
                    df_result_simple.loc[1,i] = rmse_simple
                    df_result_simple.loc[2,i] = str(round(alpha, 3)) + ' + ' + str(round(beta, 3)) + "*"+ selection



        except:
            messagebox.showerror("showerror", message= "Пожалуйста выберите показатель")
            frame_linear_1.destroy()

    combobox.bind("<<ComboboxSelected>>", selected)

    frame_simple = tk.Frame(f.second_frame,bg = 'white')
    frame_simple.pack(fill ='both',expand = True)

  

    label_result_simple = tk.Label(frame_simple, text = 'Результат сравнения простых регрессиональнных модели',font = font_title , bg = 'white')
    label_result_simple.pack(fill = 'both',expand= True,pady = (30,0))

    button_simple = tk.Button(frame_simple,text = 'Cравнение простых регрессиональнных модели', font = ('Arial,15'),bg ='white')
    button_simple.pack(pady = (30,20))

    frame_result_simple = tk.Frame(frame_simple,bg = 'white')
    #kq chi day du khi da thuc hien selected tat ca cac bien doc lap trong combobox
    def model_simple():
        global frame_result_simple
        frame_result_simple.destroy()
        

        frame_result_simple = tk.Frame(frame_simple,bg = 'white')
        frame_result_simple.pack(fill = 'both',expand= True)
        pt_result_simple = Table(frame_result_simple,dataframe = df_result_simple)

        pt_result_simple.setRowHeight(40)
        
        pt_result_simple.show()

    button_simple.config(command = model_simple)
    
    
    





    # Множественная линейная регрессия

    frame_multi = tk.Frame(f.second_frame, bg ='white')
    frame_multi.pack(fill= 'both',expand = True,pady =(30,0))

    label_title_multi = tk.Label(frame_multi, text = 'Множественная линейные регрессия - Mutiple Linear Regression', font = font_title, bg ='white')
    label_title_multi.pack(pady = (0,20), fill= 'both',expand = True)

   
    label_summary_multi = tk.Label(frame_multi,bg ='white')

    X_multi = sm.add_constant(Z)
    y_multi = Y # y из variable
    model_multi = sm.OLS(y_multi, X_multi).fit() 
    results_multi= model_multi.summary()
    plt.rc('figure', figsize=(5, 5.5))
    plt.figure().set_figwidth(9)
    plt.text(0, 0, str(results_multi), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results_multi.png')

    #форма уравнения
    coef_multi =list(model_multi.params)
    equation_multi = f'{coef_multi[0]:.2f} + {coef_multi[1]:.2f}*x1 + {coef_multi[2]:.2f} * x2 + {coef_multi[3]:.2f} * x3 + {coef_multi[4]:.2f} * x4 + {coef_multi[5]:.4f} * x5 + {coef_multi[6]:.2f} * x6'
    ypred = model_multi.predict(X_multi)
    rmse_multi = rmse(y_multi, ypred)
    r2_multi = model_multi.rsquared
    text_multi = 'R2 (R-квадрат) = ' + str(r2_multi) + "\n \n" + 'Среднеквадратическое отклонение RMSE = '+ str(rmse_multi) +  "\n \n" + 'Форма уравнения y = ' + equation_multi


    label_multi = tk.Label(frame_multi,bg = 'white', font = ('Arial',12),text = text_multi)
    label_multi.pack(fill= 'both',expand=True)

    

    #результат 

    test_summary_multi = ImageTk.PhotoImage(Image.open("results_multi.png"))

    label_summary_multi = tk.Label(frame_multi,image=test_summary_multi, bg ='white')

    label_summary_multi.image = test_summary_multi

    label_summary_multi.pack(fill= 'both',expand=True)
    
    



    # VIF de xac dinh da cong tuyen trong mo hinh ols


    # VIF , чтобы определить возникать мультиколлинеарности  или нет

    vif_data = pd.DataFrame()
    vif_data["Показатель"] = Z.columns
    vif_data["VIF"] = [variance_inflation_factor(Z.values, i)
                          for i in range(len(Z.columns))]

    label_vif= tk.Label(f.second_frame, text = 'Коэффициент инфляции дисперсии в модель с обычный метод наименьших квадратов - Variance inflation factor',font = ('Arial',15) , bg = 'white')
    label_vif.pack(fill = 'both',expand= True)

    frame_vif = tk.Frame(f.second_frame,bg ='white')
    frame_vif.pack(pady = 20)

    pt_vif = Table(frame_vif, dataframe=vif_data, width = 500)
    pt_vif.show()



    label_title_ridge = tk.Label(f.second_frame, text = 'Гребневая регрессия - Ridge Regression', font = font_title, bg = 'white')
    label_title_ridge.pack(pady = 30, fill ='both',expand = True)
    
    #Проверить VIF в Ridge 
    df_reg1 = df_full.dropna()
    columns_name = list(df_reg1.columns[1:])
    columns_name.insert(0,'alpha')
    # Create an empty data frame
    df_vif1 = pd.DataFrame(columns = columns_name)
    df_independent = df_reg1.drop(columns = 'Инфляция, дефлатор ВВП,%')
    alphas = [0.0001,0.001,0.01,0.1,10,100]
    df_vif1['alpha']= alphas
    list_rmse = []
    j=0
    for alpha in alphas:
    
        for i in df_independent.columns:
            x = df_independent.drop(columns = i, axis=1).values
            y = df_independent[i].values
            x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
            model = Ridge(alpha = alpha).fit(x_train,y_train)
            r2 = model.score(x_train,y_train)
            vif = 1/(1-r2)
            
            df_vif1.loc[j,i]='{:.8f}'.format(vif)
        X_ridge = df_reg1.drop(columns = 'Инфляция, дефлатор ВВП,%', axis=1).values
        Y_ridge = df_reg1['Инфляция, дефлатор ВВП,%'].values
        X_train,X_test,Y_train,Y_test = train_test_split(X_ridge,Y_ridge,random_state=0)
        ridge = Ridge(alpha = alpha).fit(X_train,Y_train)
        pred_ridge= ridge.predict(X_test)
        
        rmse_ridge = np.sqrt(mean_squared_error(Y_test,pred_ridge))
        list_rmse.append('{:.8f}'.format(rmse_ridge))
        j=j+1
    df_vif1['RMSE'] = list_rmse
    
    label_vif_1= tk.Label(f.second_frame, text = "Коэффициент инфляции дисперсии в гребневом регресссии c несколько alpha \n и соотвественно cреднеквадратическое отклонение (RMSE)",font = ('Arial',15) , bg = 'white')
    label_vif_1.pack(fill = 'both',expand= True)

    frame_vif_1 = tk.Frame(f.second_frame,bg ='white')
    frame_vif_1.pack(pady = 20)

    pt_vif_1 = Table(frame_vif_1, dataframe=df_vif1, width = 1000)
    pt_vif_1.show()
    

    #ridge
    
    label_ridge_alpha = tk.Label(f.second_frame, text = 'Результат в гребневой регрессии - Ridge Regression c alpha = 0.0001', font = ('Arial',15), bg = 'white')
    label_ridge_alpha.pack(pady = 30, fill ='both',expand = True)


    #X = df_reg1.drop(columns = 'Инфляция, дефлатор ВВП,%', axis=1).values
    #y = df_reg1['Инфляция, дефлатор ВВП,%'].values
    


    X = Z.values
    Y= Y.values
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)
    model_ridge = Ridge(alpha = 0.0001).fit(X_train,y_train)
    ypred_ridge= model_ridge.predict(X_test)
    rmse_ridge = np.sqrt(mean_squared_error(y_test,ypred_ridge))
    r2_score = model_ridge.score(X_train,y_train)

    equation_ridge = f'{model_ridge.intercept_:.2f} + {model_ridge.coef_[0]:.2f}*x1 + {model_ridge.coef_[1]:.2f} * x2 + {model_ridge.coef_[2]:.2f} * x3 + {model_ridge.coef_[3]:.2f} * x4 + {model_ridge.coef_[4]:.4f} * x5 + {model_ridge.coef_[5]:.2f} * x6'

    text_ridge = 'R2 (R-квадрат) = ' + str(r2_score) + '\n \n' + 'Cреднеквадратическая ошибка RMSE = ' + str(rmse_ridge) + '\n \n' + 'Формула уравнения : y = '+ equation_ridge


    label_ridge = tk.Label(f.second_frame, text = text_ridge, font = ('Arial',12), bg ='white')
    label_ridge.pack(pady = (0,20), fill= 'both',expand = True)

#vif Lasso
    


    label_title_lasso = tk.Label(f.second_frame, text = 'Регрессия Лассо - Lasso Regression', font = font_title, bg = 'white')
    label_title_lasso.pack(pady = 30 , fill ='both',expand = True)

    #Проверить VIF в Lasso 
    columns_name = list(df_reg1.columns[1:])
    columns_name.insert(0,'alpha')
    # Create an empty data frame
    df_vif_lasso = pd.DataFrame(columns = columns_name)
    df_vif_lasso['alpha']= alphas
    list_rmse_lasso = []
    j=0
    for alpha in alphas:
    
        for i in df_independent.columns:
            x_lasso = df_independent.drop(columns = i, axis=1).values
            y_lasso= df_independent[i].values
            x_train,x_test,y_train,y_test = train_test_split(x_lasso,y_lasso,random_state=0)
            model_l = Lasso(alpha = alpha).fit(x_train,y_train)
            r2 = model_l.score(x_train,y_train)
            vif = 1/(1-r2)
            
            df_vif_lasso.loc[j,i]='{:.8f}'.format(vif)
        X_lasso = df_reg1.drop(columns = 'Инфляция, дефлатор ВВП,%', axis=1).values
        Y_lasso = df_reg1['Инфляция, дефлатор ВВП,%'].values
        X_train,X_test,Y_train,Y_test = train_test_split(X_lasso,Y_lasso,random_state=0)
        lasso = Lasso(alpha = alpha).fit(X_train,Y_train)
        pred_lasso= lasso.predict(X_test)
        
        rmse_lasso = np.sqrt(mean_squared_error(Y_test,pred_lasso))
        list_rmse_lasso.append('{:.8f}'.format(rmse_lasso))
        j=j+1
    df_vif_lasso['RMSE'] = list_rmse_lasso
    
    label_vif_2= tk.Label(f.second_frame, text = "Коэффициент инфляции дисперсии в Лассо c несколько alpha \n и соотвественно cреднеквадратическое отклонение (RMSE)",font = ('Arial',15) , bg = 'white')
    label_vif_2.pack(fill = 'both',expand= True)

    frame_vif_2 = tk.Frame(f.second_frame,bg ='white')
    frame_vif_2.pack(pady = 20)

    pt_vif_2 = Table(frame_vif_2, dataframe=df_vif_lasso, width = 1000)
    pt_vif_2.show()
    

    #Lasso
    
    label_lasso_alpha = tk.Label(f.second_frame, text = 'Результат в регрессии Лассо - Lasso Regression c alpha = 0.0001', font = ('Arial',15), bg = 'white')
    label_lasso_alpha.pack(pady = 30, fill ='both',expand = True)


    #X = df_reg1.drop(columns = 'Инфляция, дефлатор ВВП,%', axis=1).values
    #y = df_reg1['Инфляция, дефлатор ВВП,%'].values
    
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)
    model_lasso = Lasso(alpha = 0.0001).fit(X_train,y_train)
    ypred_lasso = model_lasso.predict(X_test)
    rmse_lasso = np.sqrt(mean_squared_error(y_test,ypred_lasso))
    r2_score_lasso = model_lasso.score(X_train,y_train)

    equation_lasso = f'{model_lasso.intercept_:.2f} + {model_lasso.coef_[0]:.2f}*x1 + {model_lasso.coef_[1]:.2f} * x2 + {model_lasso.coef_[2]:.2f} * x3 + {model_lasso.coef_[3]:.2f} * x4 + {model_lasso.coef_[4]:.4f} * x5 + {model_lasso.coef_[5]:.2f} * x6'

    text_lasso = 'R2 (R-квадрат) = ' + str(r2_score_lasso) + '\n \n' + 'Cреднеквадратическая ошибка RMSE = ' + str(rmse_lasso) + '\n \n' + 'Формула уравнения : y = '+ equation_lasso

    
    label_lasso = tk.Label(f.second_frame, text = text_lasso, font = ('Arial',12), bg ='white')
    label_lasso.pack(pady = (0,20), fill= 'both',expand = True)



    

    #результат сравнения множественные регрессиональнной модель
    
    data = {'Показатель':['R2 (R-квадрат)','Cреднеквадратическая ошибка RMSE','Формула уравнения'],
            'Множественная линейная регрессия':[r2_multi,rmse_multi,equation_multi],
            'Гребневая регрессия':[r2_score,rmse_ridge,equation_ridge],
            'Регрессия Лассо':[r2_score_lasso,rmse_lasso,equation_lasso]}
    
    df_result = pd.DataFrame(data = data)

    label_result = tk.Label(f.second_frame, text = 'Результат сравнения множественных регрессиональнных модели',font = font_title , bg = 'white')
    label_result.pack(fill = 'both',expand= True, pady=30)

    frame_result = tk.Frame(f.second_frame,bg = 'white')
    frame_result.pack(fill = 'both',expand= True)
    pt_result = Table(frame_result,dataframe = df_result)

    pt_result.setRowHeight(40)

    pt_result.show()

    #таблица по инфляции по дефлятор по годом 

    df_deflator_1 = df_deflator.reset_index()
    df_deflator_1 = df_deflator_1.sort_values("Country")
    label_country_1 = tk.Label(f.second_frame, text = 'Таблица по инфляции, дефлятор ВВП 2000-2021 всех странах',font = font_title, bg = 'white')

    label_country_1.pack(fill = 'both',expand= True, pady = 30)

    
    
    df_deflator_1=df_deflator_1.dropna()
    #dataframe для создание регрессионная модель 
    df_deflator_1['Средние'] = df_deflator_1.iloc[:,1:].mean(axis=1)
    df_deflator_1['Cтандартное отклонение'] = df_deflator_1.iloc[:,1:].std(axis=1)
    df_deflator_1['Средние изменение'] = (df_deflator_1['YR2021'] / df_deflator_1['YR2000'])**(1/20)
    df_deflator_1['Максимум'] = df_deflator_1.iloc[:,1:].max(axis=1)
    df_deflator_1['Минимум'] = df_deflator_1.iloc[:,1:].min(axis=1)

    list_columns = ['Country',"Средние",'Cтандартное отклонение','Средние изменение','Максимум','Минимум']
    
    df_deflator_2 = df_deflator_1[list_columns]
    
    list_columns.pop(0)
    df_deflator_1 = df_deflator_1.drop(columns = list_columns)
    frame_country_1 = tk.Frame(f.second_frame,bg = 'white')
    frame_country_1.pack(fill = 'both',expand= True)

    pt_country = Table(frame_country_1, dataframe=df_deflator_1, showtoolbar=True, showstatusbar=True)
    
    pt_country.setRowHeight(30)
    pt_country.show()

    label_country_2 = tk.Label(f.second_frame, text = 'Cредние, cтандартное отклонение, средние изменение, максимум, минимум',font = font_title , bg = 'white')
    label_country_2.pack(fill = 'both',expand= True,pady=30)

    frame_country_2 = tk.Frame(f.second_frame,bg = 'white')
    frame_country_2.pack()
    
    pt_country_2 = Table(frame_country_2, dataframe = df_deflator_2, width = 800,showtoolbar=True, showstatusbar=True)
    
    pt_country_2.setRowHeight(30)
    pt_country_2.show()
    #Сравнение инфляции, дефлятор ВВП и CPI


    frame_country_3 = tk.Frame(f.second_frame, bg = 'white')
    frame_country_3.pack(fill='both',expand=True)

    label_country_1 = tk.Label(frame_country_3, text = 'Сравнение инфляции, дефлятор ВВП и ИПЦ страны \n и показывание регрессионной модель инфляции страны в 2000-2021 году',bg = 'white', font = font_title)
    label_country_1.pack(fill='both',expand=True, pady=(30,0))

    frame_country_4 = tk.Frame(frame_country_3, bg = 'white')
    frame_country_4.pack(pady = 20, fill='both',expand=True)


    label_country_2 = tk.Label(frame_country_4, text = 'Ввести страны',bg = 'white',font = ('Arial',12))
    label_country_2.grid(padx=(screen_width/2-200,20), sticky='w',column=0,row=0)
    entry_country = tk.Entry(frame_country_4, bg = 'white')
    entry_country.grid(sticky= 'e',column=1,row=0)

    button_country = tk.Button(frame_country_3, text = 'Загрузка динамики инфляции', font =('Arial',12))
    button_country.pack(pady=20)

    label_country = tk.Label(frame_country_3,bg = 'white')
    label_dynamic = tk.Label(frame_country_3)

    label_dynamic_cpi = tk.Label(frame_country_3)
    def country():
        try:
            global entry_country, label_country,label_dynamic,label_dynamic_cpi
            
            label_country.destroy()
            label_dynamic.destroy()
            label_dynamic_cpi.destroy()

            country = entry_country.get()
            cols = df_cpi.columns
            data_cpi = df_cpi.loc[country]
            data_deflator= df_deflator.loc[country]
            text_deflator = [str('{:.2f}'.format(x)) for x in list(data_deflator)] 
            text_cpi = [str('{:.2f}'.format(x)) for x in list(data_cpi)]
            fig_country = go.Figure()
            # Create and style traces
            fig_country.add_trace(go.Scatter(x=cols, y=data_cpi, name='ИПЦ', 
                                text = text_cpi, 
                                textfont=dict(color='red'),
                                mode='lines+markers+text',
                                marker=dict(color='#5D69B1', size=8),
                                textposition="bottom center",
                                line=dict(color='firebrick', width=2, dash='dot')))
            fig_country.add_trace(go.Scatter(x=cols, y=data_deflator, name = 'Инфляция, дефлятор ВВП',
                                text = text_deflator,
                                textfont=dict(color='#E58606'),
                                mode='lines+markers+text',
                                marker=dict(color='#5D69B1', size=8),
                                textposition="top center",
                                line=dict(color='royalblue', width=4)))
            fig_country.update_layout(
                    title='Инфляции, дефлятор ВВП и ИПЦ ' + country,
                    xaxis_title='Год',
                    yaxis_title='Значение',
                    width = 1200, height = 500,
                    margin=dict(l=20, r=20, t=100, b=0),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                    
            fig_country.write_image("fig_country.png", engine="kaleido")
            test_country = ImageTk.PhotoImage(Image.open("fig_country.png"))
            label_country = tk.Label(frame_country_3,image=test_country, bg ='white')
            label_country.image = test_country
            label_country.pack(fill= 'both',expand=True)
            
            
            

            # Регрессионный модель инфляции (дефлятор ВВП) по времени 
            
            fig_dynamic = px.scatter( x=list(range(2000,2022)), y=data_deflator, trendline="ols")
            fig_dynamic.update_layout( margin=dict(l=0, r=0, t=50, b=0),
                            legend=dict(orientation="h",yanchor="bottom",
                            y=0.99,xanchor="right", x=1))
            model = px.get_trendline_results(fig_dynamic)
            r2_rsquared = model.px_fit_results.iloc[0].rsquared
            alpha = model.iloc[0]["px_fit_results"].params[0]
            beta = model.iloc[0]["px_fit_results"].params[1]
            fig_dynamic.data[1].line.color = 'red'
            fig_dynamic.data[0].name = 'Наблюдение'
            fig_dynamic.data[0].showlegend = True
            fig_dynamic.data[1].name = '\n'+'Инфляция, дефлатор ВВП = ' + str(round(alpha, 3)) + ' + ' + str(round(beta, 3)) + "*"+ 'x'
            fig_dynamic.data[1].showlegend = True
            fig_dynamic.add_trace(go.Scatter(x=list(range(2000,2022)), y=data_deflator,
                         name = "R-квадрат" + ' = ' + str(r2_rsquared),
                         showlegend=True,
                         mode='markers',
                         marker=dict(color='rgba(0,0,0,0)')
                         ))
            fig_dynamic.add_trace(go.Scatter(x=list(range(2000,2022)), y=data_deflator, name = 'Инфляция, дефлятор ВВП',
                                text = text_deflator,
                                textfont=dict(color='#E58606'),
                                mode='lines+markers+text',
                                marker=dict(color='#5D69B1', size=8),
                                textposition="top center",
                                line=dict(color='royalblue', width=4)))
            fig_dynamic.update_layout(
                    title='Регрессионный модель инфляции (дефлятор ВВП) по времени стрнана ' +country,
                    xaxis_title='Год',
                    yaxis_title='Инфляция, дефлятор ВВП',
                    width = 1200, height = 500,
                    margin=dict(l=20, r=20, t=100, b=0))
            fig_dynamic.write_image("fig_dynamic.png", engine="kaleido")
            test_dynamic = ImageTk.PhotoImage(Image.open("fig_dynamic.png"))
            label_dynamic = tk.Label(frame_country_3,image=test_dynamic, bg ='white')
            label_dynamic.image = test_dynamic
            label_dynamic.pack(fill= 'both',expand=True)
            
            
             # Регрессионный модель индекс потребительских цен по времен
            fig_dynamic_cpi = px.scatter( x=list(range(2000,2022)), y=data_cpi, trendline="ols")
            fig_dynamic_cpi.update_layout( margin=dict(l=0, r=0, t=50, b=0),
                            legend=dict(orientation="h",yanchor="bottom",
                            y=0.99,xanchor="right", x=1))
            model_cpi = px.get_trendline_results(fig_dynamic_cpi)
            r2_rsquared = model_cpi.px_fit_results.iloc[0].rsquared
            alpha = model_cpi.iloc[0]["px_fit_results"].params[0]
            beta = model_cpi.iloc[0]["px_fit_results"].params[1]
            fig_dynamic_cpi.data[1].line.color = 'red'
            fig_dynamic_cpi.data[0].name = 'Наблюдение'
            fig_dynamic_cpi.data[0].showlegend = True
            fig_dynamic_cpi.data[1].name = '\n'+'Индекс потребительских цен = ' + str(round(alpha, 3)) + ' + ' + str(round(beta, 3)) + "*"+ 'x'
            fig_dynamic_cpi.data[1].showlegend = True
            fig_dynamic_cpi.add_trace(go.Scatter(x=list(range(2000,2022)), y=data_cpi,
                         name = "R-квадрат" + ' = ' + str(r2_rsquared),
                         showlegend=True,
                         mode='markers',
                         marker=dict(color='rgba(0,0,0,0)')
                         ))
            fig_dynamic_cpi.add_trace(go.Scatter(x=list(range(2000,2022)), y=data_cpi, name = 'Индекс потребительских цен - ИПЦ',
                        text = text_cpi, 
                        textfont=dict(color='red'),
                        mode='lines+markers+text',
                        marker=dict(color='#5D69B1', size=8),
                        textposition="bottom center",
                        line=dict(color='firebrick', width=2, dash='dot')))
            fig_dynamic_cpi.update_layout(
                    title='Регрессионный модель ИПЦ по времени страна ' + country,
                    xaxis_title='Год',
                    yaxis_title='Индекс потребительских цен - ИПЦ',
                    width = 1200, height = 500,
                    margin=dict(l=20, r=20, t=100, b=0))
            fig_dynamic_cpi.write_image("fig_dynamic_cpi.png", engine="kaleido")
            test_dynamic_cpi = ImageTk.PhotoImage(Image.open("fig_dynamic_cpi.png"))
            label_dynamic_cpi = tk.Label(frame_country_3,image=test_dynamic_cpi, bg ='white')
            label_dynamic_cpi.image = test_dynamic_cpi
            label_dynamic_cpi.pack(fill= 'both',expand=True)
            
            #entry_country.delete(0,'end')

        except:
            messagebox.showerror("showerror", message= "Неверное входное значение")
            
    button_country.config(command =country)


    #print(np.argmax(df_deflator['YR2007']))
    #Рисуем в первое окно c характеристками распределения

    frame_distribution = tk.Frame(f.second_frame, bg = 'white')
    frame_distribution.pack(pady = (0,30),fill= 'both', expand = True)

    label_distribution = tk.Label(frame_distribution, text = 'Характеристки распределение (2000-2021)', font = font_title, bg = 'white')
    label_distribution.pack(fill= 'both', expand = True)
    
    frame_distribution_1 = tk.Frame(frame_distribution, bg = 'white')
    frame_distribution_1.pack(pady =(30,0),fill= 'both', expand = True)

    label_distribution_1 = tk.Label(frame_distribution_1, text = 'Bвести год', font = ('Arial',12), bg ='white')
    label_distribution_1.grid(padx=(screen_width/2-200,20),column = 0, row=0,sticky='w')
    entry_distribution = tk.Entry(frame_distribution_1)
    entry_distribution.grid(column = 1, row=0,sticky = 'e')

    label_distribution_2 = tk.Label(frame_linear_1, bg ='white')


    '''fig = plt.figure(figsize=(12,5))

            sns.histplot(df_deflator,x = year, kde = True)
            plt.title('Характеристки распделения ' + entry_distribution.get() + ' году' )
            plt.xlabel('Инфляции, дефлятор ВВП')'''

    
    def plot():
        global label_distribution_2
        label_distribution_2.destroy()
        year = 'YR'+ entry_distribution.get()
        if year in list(df_deflator.columns):
            
            mean = df_deflator[year].mean()
            median  = df_deflator[year].median()
            quantile_025 = np.nanquantile(df_deflator[year],0.25)
            quantile_075 = np.nanquantile(df_deflator[year],0.75)
            iqr = quantile_075-quantile_025
            min_bp = quantile_025 - 1.5*iqr
            max_bp = quantile_075 + 1.5*iqr
            plt.figure(clear=True)
            fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2,2,figsize=(15,6))
            
            sns.histplot(df_deflator,x = year , ax = ax1, kde = True)
            ax1.axvline(x=mean, label='Инфляции, дефлятор ВВП', c='red')
            ax1.axvline(x=median, label='Инфляции, дефлятор ВВП', c='green')
            ax1.set(title = 'Характеристки распределение ' + entry_distribution.get() + ' году', xlabel = 'Инфляции, дефлятор ВВП')
            y_min, y_max = ax1.get_ylim()
            ax1.text(mean,y_max-10,'mean = ' + f'{mean:.2f}', color = 'red')
            ax1.text(median,y_max-5,'median = '+ f'{median:.2f}', color ='green')
           
            sns.boxplot(df_deflator, x= year,ax = ax2)
            ax2.set(title = 'Характеристки распределение  ' + entry_distribution.get() + ' году', xlabel = 'Инфляции, дефлятор ВВП')
            y_min, y_max = ax2.get_ylim()
            ax2.text(quantile_025-3,y_min-0.05,'Q1 = '+ f'{quantile_025:.2f}')
            ax2.text(quantile_075,y_max+0.1,'Q3 = '+ f'{quantile_075:.2f}')
            ax2.text(min_bp,y_min-0.25,'Min = '+ f'{min_bp:.2f}')
            ax2.text(max_bp,y_max+0.25,'Max = '+ f'{max_bp:.2f}')
            
            mean_cpi = df_cpi[year].mean()
            median_cpi = df_cpi[year].median()
            quantile_025_cpi = np.nanquantile(df_cpi[year],0.25)
            quantile_075_cpi = np.nanquantile(df_cpi[year],0.75)
            iqr_cpi = quantile_075_cpi-quantile_025_cpi
            min_bp_cpi = quantile_025_cpi - 1.5*iqr_cpi
            max_bp_cpi = quantile_075_cpi + 1.5*iqr_cpi
            sns.histplot(df_cpi,x = year , ax = ax3, kde = True)
            ax3.axvline(x=mean_cpi, c='red')
            ax3.axvline(x=median_cpi, c='green')
            
            ax3.set( xlabel = 'Индекс потребительских цен - ИПЦ')
            y_min, y_max = ax3.get_ylim()
            ax3.text(mean_cpi,y_max-10,'mean = ' + f'{mean_cpi:.2f}', color = 'red')
            ax3.text(median_cpi,y_max-5,'median = '+ f'{median_cpi:.2f}', color ='green')
           
            sns.boxplot(df_cpi, x= year,ax= ax4)
            ax4.set( xlabel = 'Индекс потребительских цен - ИПЦ')
            y_min, y_max = ax4.get_ylim()
            ax4.text(quantile_025_cpi-3,y_min-0.05,'Q1 = '+ f'{quantile_025_cpi:.2f}')
            ax4.text(quantile_075_cpi,y_max+0.1,'Q3 = '+ f'{quantile_075_cpi:.2f}')
            ax4.text(min_bp_cpi,y_min-0.25,'Min = '+ f'{min_bp_cpi:.2f}')
            ax4.text(max_bp_cpi,y_max+0.25,'Max = '+ f'{max_bp_cpi:.2f}')
            plt.savefig('fig_distribution.png')
            
            test_distribution = ImageTk.PhotoImage(Image.open("fig_distribution.png"))
            label_distribution_2 = tk.Label(frame_distribution,image=test_distribution, bg ='white')
            label_distribution_2.image = test_distribution
            label_distribution_2.pack(fill= 'both',expand=True)
        else:
            messagebox.showerror("showerror", message= "Неверное входное значение")
    button_distribution= tk.Button(frame_distribution, text = 'Характеристики распделения', command = plot, font = ('Arial',12))
    button_distribution.pack(pady =(20,0))

    #топ 15 высокий

    frame_top = tk.Frame(f.second_frame, bg='white')
    frame_top.pack(fill = 'both',expand= True)

    label_top_1 = tk.Label(frame_top, text = "Топ 15 странах с высокими и нижкими инфляциями в году (2000-2021)", bg = 'white', font = font_title)
    label_top_1.pack(pady = 10, fill = 'both',expand= True)

    frame_top_1 = tk.Frame(frame_top, bg='white')
    frame_top_1.pack(fill = 'both',expand= True)

    label_top_2 = tk.Label(frame_top_1,text = "Ввести год", font = ('Arial',12), bg='white')
    label_top_2.grid(padx = (screen_width/2-100,20), column=0, row = 0, sticky='e')

    entry_top = tk.Entry(frame_top_1, bg = 'yellow')
    entry_top.grid(column=1, row = 0, sticky='w')
    #frame chua 2 anh cua max and min

    button_top= tk.Button(frame_top, text = 'Загрузка результат визуализации ', font =('Arial',12))
    button_top.pack(pady=20)

    frame_child= tk.Frame(frame_top, bg ='white')
    frame_child.pack(fill='both', expand =True) 

    label_top = tk.Label(frame_child)
    label_short = tk.Label(frame_child)
    label_top_cpi = tk.Label(frame_child)
    label_short_cpi = tk.Label(frame_child)
    
    def top():
        try:
            global label_top, frame_child,label_short,label_top_cpi,label_short_cpi
            #Define a function to check if a frame_child exists or not
            exists = frame_child.winfo_exists() 
            if exists == 0:
                frame_child= tk.Frame(frame_top, bg ='white')
                frame_child.pack(fill='both', expand =True) 
            label_top.destroy()
            label_short.destroy()
            label_top_cpi.destroy()
            label_short_cpi.destroy()
            year = 'YR' + entry_top.get()
            top_15 = df_deflator.sort_values(year,ascending = False)[[year]].head(15)
            top_15_cpi= df_cpi.sort_values(year,ascending = False)[[year]].head(15)
            fig1= px.bar(top_15, y= top_15.index,x = year,color = year,
                        text_auto = '.3s', orientation = 'h',
                        title = 'Страны с высокими инфляциями (дефлятор ВВП) в ' + entry_top.get() + ' году')
            fig1.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        yaxis = {'categoryorder':'total ascending'})
            fig1.write_image("fig_top.png", engine="kaleido")
            fig3= px.bar(top_15_cpi, y= top_15_cpi.index,x = year,color = year,
                        text_auto = '.3s', orientation = 'h',
                        title = 'Страны с высокими ИПЦ в ' + entry_top.get() + ' году')
            fig3.update_layout(
                        margin=dict(l=20, r=20, t=40, b=20),
                        yaxis = {'categoryorder':'total ascending'})
            fig3.write_image("fig_top_cpi.png", engine="kaleido") 
            test_top = ImageTk.PhotoImage(Image.open("fig_top.png"))
            label_top = tk.Label(frame_child,image=test_top, bg ='white')
            label_top.image = test_top
            label_top.grid(column = 0,row=0)

            test_top_cpi = ImageTk.PhotoImage(Image.open("fig_top_cpi.png"))
            label_top_cpi = tk.Label(frame_child,image=test_top_cpi, bg ='white')
            label_top_cpi.image = test_top_cpi
            label_top_cpi.grid(column=0,row=1)


            top_15_short = df_deflator.sort_values(year)[[year]].head(15)
            top_15_short_cpi = df_cpi.sort_values(year)[[year]].head(15)

            fig2= px.bar(top_15_short,y= top_15_short.index,x = year, color = year,
                        text_auto = '.3',orientation = 'h',
                        title = 'Страны с нижкими инфляциями (дефлятор ВВП) в ' + entry_top.get() + ' году')
            fig2.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis = {'categoryorder':'total descending'})
            fig2.write_image("fig_short.png", engine="kaleido") 
            fig4= px.bar(top_15_short_cpi,y= top_15_short_cpi.index,x = year, color = year,
                        text_auto = '.3',orientation = 'h',
                        title = 'Страны с нижкими ИПЦ  в ' + entry_top.get() + ' году')
            fig4.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis = {'categoryorder':'total descending'})
            fig4.write_image("fig_short_cpi.png", engine="kaleido") 
            test_short = ImageTk.PhotoImage(Image.open("fig_short.png"))
            label_short = tk.Label(frame_child,image=test_short, bg ='white')
            label_short.image = test_short
            label_short.grid(column=1, row=0)
            test_short_cpi = ImageTk.PhotoImage(Image.open("fig_short_cpi.png"))
            label_short_cpi = tk.Label(frame_child,image=test_short_cpi, bg ='white')
            label_short_cpi.image = test_short_cpi
            label_short_cpi.grid(column=1,row=1)
            #entry_top.delete(0,'end')
        except:
            messagebox.showerror("showerror", message= "Неверное входное значение")
            frame_child.destroy()

    button_top.config(command=top)


    f.root.state('zoomed')
    f.root.mainloop()