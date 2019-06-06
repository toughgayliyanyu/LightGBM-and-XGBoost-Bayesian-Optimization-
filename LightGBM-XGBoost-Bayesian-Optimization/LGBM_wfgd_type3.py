# -*- coding：utf-8 -*-
# 引入模型
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn import preprocessing

# 绘制不同参数下MSE的对比曲线
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
import matplotlib.pyplot as plt

data = np.loadtxt('data_fgd_10000_2.txt')
#data_test = np.loadtxt('data_fgd_50.txt')
X=data[:, 1:10]
y=data[:, 0]
X_train_input, X_test1_input, y_train, y_test1 = train_test_split(X, y, test_size=0.005)
data_test = np.loadtxt('data_fgd_50two.txt')
X_test_input=data_test[:, 1:10]
y_test=data_test[:, 0]
print(X_train_input.shape)
print(X_test_input.shape)
#数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train_input)
X_train1 = min_max_scaler.fit_transform(X)
X_test = min_max_scaler.fit_transform(X_test_input)
#####################
starttime = time.time()
#model = lgbm.LGBMRegressor(boosting_type='gbdt', objective='regression',num_leaves=50,
#                            learning_rate=0.15, min_data_in_leaf=200,bagging_fraction=0.8,min_child_weight=49.494858,
#                           max_depth=6,n_estimators=1000,max_bin=200,feature_fraction=0.5,min_split_gain=0.04620938,
#                            metric='rmse')
model = lgbm.LGBMRegressor(boosting_type='gbdt', objective='regression',num_leaves=50,
                            learning_rate=0.15, min_data_in_leaf=200,bagging_fraction=0.8,min_child_weight=0.001,
                           max_depth=6,n_estimators=1000,max_bin=200,feature_fraction=0.8,min_split_gain=0.04620938,
                            metric='rmse')
model.fit(X_train1, y)
endtime = time.time()
time_spend_lgbm=endtime - starttime
print('lightgbm运行时间:',time_spend_lgbm)
lgbm.plot_importance(model, max_num_features=10)#max_features表示最多展示出前10个重要性特征，可以自行设置
plt.show()
#gbdt
'''starttime = time.time()
model_gbdt = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.15,
                                  max_depth=6, random_state=0, loss='ls')
model_gbdt.fit(X_train,y_train)
endtime = time.time()
time_spend_gbdt=endtime - starttime
print('gbdt运行时间:',time_spend_gbdt)'''
#xgboost
starttime = time.time()
model_xgb = XGBRegressor(
    learn_rate = 0.15,
    max_depth = 6,
    min_child_weight = 10,
    gamma = 0,
    subsample = 0.7,
    colsample_bytree = 0.7,
    reg_alpha = 0.8,
    objective = 'reg:linear',
    n_estimators = 1000
)
model_xgb.fit(X_train,y_train)
endtime = time.time()
time_spend_xgb=endtime - starttime
print('XgBoost运行时间:',time_spend_xgb)
from xgboost import plot_importance
plot_importance(model_xgb, max_num_features=10)#max_features表示最多展示出前10个重要性特征，可以自行设置
plt.show()
#svr
#画出算法运行时间柱状图
x=[0,1]
name_list = ['lightGBM','xgboost']
num_list = [time_spend_lgbm,time_spend_xgb]
a=plt.bar(x, num_list,color='rgbc',tick_label=name_list,label='算法训练时间对比图')
plt.text(x[0], num_list[0]+0.005, '%.3f' % num_list[0], ha='center', va= 'bottom',fontsize=11)
#plt.text(x[1], num_list[1]+0.005, '%.3f' % num_list[1], ha='center', va= 'bottom',fontsize=11)
plt.text(x[1], num_list[1]+0.005, '%.3f' % num_list[1], ha='center', va= 'bottom',fontsize=11)
plt.legend()
plt.show()
# 给出训练数据的预测值
train_out = model.predict(X)
#train_out_gbdt=model_gbdt.predict(X_train)
train_out_xgb=model_xgb.predict(X_train)
# 计算MSE
train_mse = mse(y, train_out)
#train_mse_gbdt=mse(y_train,train_out_gbdt)
train_mse_xgb=mse(y_train,train_out_xgb)
print('the lightgbm train mse is:',train_mse)
#print('the gbdt train mse is:',train_mse_gbdt)
print('the xgb train mse is:',train_mse_xgb)
# 给出验证数据的预测值
print("y_test",y_test)
print("x_test",X_test)
print("x_test_len",len(X_test))
g1=[]
g2=[]
sum_g1=0
sum_g2=0
final_result=[]
error_lgbm=[]
error_xgb=[]
for i in range(len(X_test)):
    lgbm_pred_i=model.predict((X_test[i]).reshape(1, -1))
    xgb_predict_i=model_xgb.predict((X_test[i]).reshape(1, -1))
    error_lgbm=y_test[i]-lgbm_pred_i
    print(error_lgbm)
    error_xgb=y_test[i]-xgb_predict_i
    print(error_xgb)
    if i<6 :
        if ((error_lgbm>=0) and (error_xgb>=0)) or ((error_lgbm<0) and (error_xgb<0)):
            if abs(error_lgbm)<=abs(error_xgb):
                g1.append(1)
                g2.append(0)
                #final_result.append(lgbm_pred_i)
            else:
                g1.append(0)
                g2.append(1)
                #final_result.append(xgb_predict_i)
        if (error_lgbm>0 and error_xgb<0) or (error_lgbm<0 and error_xgb>0):
            g1.append(abs(error_lgbm)/(abs(error_lgbm)+abs(error_xgb)))
            g2.append(abs(error_xgb)/(abs(error_lgbm)+abs(error_xgb)))
        if len(g1)==6:
            sum_g1=sum(g1)
            sum_g2=sum(g2)
            print("sum_g1",sum_g1)
            print("sum_g2",sum_g2)
    else:
        g1.append(sum(g1[i-6:i])/6)
        g2.append(sum(g2[i - 6:i ])/6)
    final_result.append((lgbm_pred_i*g1[i])+(xgb_predict_i*g2[i]))
print(g1)
print("........")
print(g2)
print(".........")
print(final_result)

        #final_result.append((lgbm_pred_i*sum_g1/9)+(xgb_predict_i*sum_g2/9))


#final_result.append(p1*lgbm_pred_i+p2*xgb_predict_i)
#print("the final result is:",np.array(final_result))
    #error_lgbm.append(y_test[i]-lgbm_pred_i)
    #error_xgb.append(y_test[i]-xgb_predict_i)
#print("error_lgbm",np.array(error_lgbm))
#print("error_xgb",np.array(error_xgb))
add_yan = model.predict(X_test)
#gbdt_predict=model_gbdt.predict(X_test)
xgb_predict=model_xgb.predict(X_test)
raler_up=[y_test[i]-add_yan[i] for i in range(len(X_test))]
sum_raler_up=sum(raler_up)/49
print("the sum_raler_up",sum_raler_up)
#raler_up_gbdt=[y_test[i]-gbdt_predict[i] for i in range(len(X_test))]
#sum_raler_up_gbdt=sum(raler_up_gbdt)/49
#print("the sum_raler_up_gbdt",sum_raler_up_gbdt)
raler_up_xgb=[y_test[i]-xgb_predict[i] for i in range(len(X_test))]
sum_raler_up_xgb=sum(raler_up_xgb)/49
print("the sum_raler_up_xgb",sum_raler_up_xgb)
raler_up_comp=[y_test[i]-final_result[i] for i in range(len(X_test))]
sum_raler_up_comp=sum(raler_up_comp)/49
print("the sum_raler_up_comp",sum_raler_up_comp)
func = lambda x,y:x/y
result = map(func,raler_up,y_test)
list_result = list(result)
print('the lightgbm rellative error is:',list_result)
print('###############')
#func_gbdt = lambda x,y:x/y
#result_gbdt = map(func,raler_up_gbdt,y_test)
#list_result_gbdt = list(result_gbdt)
#print('the gbdt rellative error is:',list_result_gbdt)
#print('###############')
func_xgb = lambda x,y:x/y
result_xgb = map(func,raler_up_xgb,y_test)
list_result_xgb = list(result_xgb)
print('the xgb rellative error is:',list_result_xgb)
print('###############')
func_xgb = lambda x,y:x/y
result_comp = map(func,raler_up_comp,y_test)
list_result_comp = list(result_comp)
print('the comp rellative error is:',list_result_comp)
print('###############')
print( [x*100 for x in list_result] )
#print( [x*100 for x in list_result_gbdt] )
print( [x*100 for x in list_result_xgb] )
print( [x*100 for x in list_result_comp] )
#plt.plot(list(range(len(list_result))), [x*100 for x in list_result],'b--',marker='s',label='relative error of lightgbm')
#plt.plot(list(range(len(list_result_gbdt))), [x*100 for x in list_result_gbdt],'r--',marker='o',label='relative error of gbdt')
#plt.plot(list(range(len(list_result_xgb))), [x*100 for x in list_result_xgb],'g--',marker='*',label='relative error of xgb')
plt.xlabel("n",fontsize=14)
plt.ylabel("相对误差/%",fontsize=14)
plt.scatter(list(range(len(list_result_comp))), [x*100 for x in list_result_comp],c='r',marker='o')
plt.legend()
plt.show()
#print(yuce_re)
# 计算MSE
#add_mse = mse(yanzhgdata[:, 0], add_yan)
add_mse = mse(y_test, add_yan)
#add_mse_gbdt=mse(y_test,gbdt_predict)
add_mse_xgb=mse(y_test,xgb_predict)
add_mse_compen=mse(y_test,final_result)
print('the lightgbm test mse is:',add_mse)
#print('the gbdt test mse is:',add_mse_gbdt)
print('the xgb test mse is:',add_mse_xgb)
print('the compent test mse is:',add_mse_compen)
#plt.figure(figsize=(17, 9))
plt.xlabel("n",fontsize=14)
plt.ylabel("脱硫效率/%",fontsize=14)
plt.plot(list(range(len(y_test))), y_test,c='r',linestyle='-',marker='o',linewidth=1.5,label='真实值')
#plt.plot(list(range(len(add_yan))), add_yan, c='b',linestyle='-.',marker='s', linewidth=1.5,label='predict lightgbm')
#plt.plot(list(range(len(yucede_se_gbdt))), yucede_se_gbdt, 'black',marker='h', label='predict gbdt')
#plt.plot(list(range(len(xgb_predict))), xgb_predict,c='g',linestyle='--',marker='*', linewidth=1.5,label='predict xgb')
plt.plot(list(range(len(final_result))), final_result, c='k',linestyle=':',marker='h',linewidth=1.5, label='预测值')
#plt.xlim(-1, 100 + 1)
plt.legend()
plt.show()
#return train_mse, add_mse
