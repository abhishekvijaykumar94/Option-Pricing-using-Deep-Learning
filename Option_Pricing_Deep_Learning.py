import streamlit as st
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed,Lambda, Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import scipy.stats as si
import time
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

st.title("Deep Learning Based Pricing Test")

class Derivative:

    def __init__(self):
        self.Stock_Price = st.number_input("Enter the Stock Price:", value=1.0, key="St", min_value=0.5, max_value=1.5)
        self.Strike_Price = st.number_input("Enter the Strike Price:", key="K", value=1.0, min_value=1.0, max_value=1.0)
        self.Time_to_Maturity = st.number_input("Enter the Time to Maturity:", key="T2M", value=1.0, min_value=0.01,max_value=1.00)
        self.Volatility = st.number_input("Enter the Volatility:", key="Vol", value=0.2, min_value=0.01, max_value=0.45)
        self.Risk_free_Rate = st.number_input("Enter the Risk free rate: ", key="Rf", value=0.05, min_value=0.0,max_value=0.14)

    def Simulated_Price_Paths(self,S0, K, r, sigma, T, n, Numer_of_Periods):
        n=self.Number_of_Paths
        delta_T = 1.0 / Numer_of_Periods
        self.Simulated_Prices = np.zeros((Numer_of_Periods, n))
        self.Simulated_Prices_Antithetic = np.zeros((Numer_of_Periods, n))
        f1 = (r - 0.5 * sigma ** 2) * delta_T
        f2 = sigma * np.sqrt(delta_T)
        z = np.random.normal(size=(Numer_of_Periods, n))
        z_A = -z
        self.Simulated_Prices[:1, ] = S0 * np.exp(f1 + f2 * z[1, :])
        self.Simulated_Prices_Antithetic[:1, ] = S0 * np.exp(f1 + f2 * z_A[1, :])
        for j in range(Numer_of_Periods - 1):
            self.Simulated_Prices[j + 1, :] = self.Simulated_Prices[j, :] * np.exp(f1 + f2 * z[(j + 1), :])
            self.Simulated_Prices_Antithetic[j + 1, :] = self.Simulated_Prices_Antithetic[j, :] * np.exp(f1 + f2 * z_A[(j + 1), :])

    def Compile_Neural_Network(self,path):
        self.model = Sequential()
        self.model.add(Dense(400, input_dim=self.x_train.shape[1], activation='relu'))
        self.model.add(Dense(400, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        adam = Adam(lr=0.005, decay=1e-6)
        self.model.compile(loss='mse', optimizer=adam, metrics=['mse'])
        self.model.load_weights(path)

    def Predict_Neural_Network(self):
        self.start = time.time()
        self.prediction =np.array( self.model.predict(self.x_train) - 1)
        self.stop = time.time()

    def print_results(self):
        st.write("Here he we can compare the prices, time taken as well as the error in pricing expressed as error in dollars on a million dollar notional")
        self.NN_Option_Price = np.maximum(round(float(self.prediction), 6), 0)
        self.NN_Time_Taken = round(float(self.stop - self.start) * 1000, 0)
        self.NN_Error = int(((self.MC_Option_Price - self.NN_Option_Price) * 1000000))
        Derivative_Pricing_Data = [(self.MC_Option_Price, self.MC_Time_Taken,self.MC_Stderr), (self.NN_Option_Price, self.NN_Time_Taken, self.NN_Error)]
        Derivative_Pricing_Data = pd.DataFrame(Derivative_Pricing_Data, columns=['Option Price', 'Time taken in Miliseconds', 'Pricing Error per mil$'],index=['Closed Form', 'Neural Network'])
        st.table(Derivative_Pricing_Data)

    def speed_accuracy_test(self,path_x_train,path_y_train):
        h5f = h5py.File(path_x_train)
        self.x_train = h5f['dataset_1'][:]
        #st.write("Shape:",self.x_train.shape)
        h5f = h5py.File(path_y_train)
        self.y_train = h5f['dataset_1'][:]
        self.Predict_Neural_Network()
        self.prediction=np.squeeze(self.prediction)
        self.average_error= round((np.mean(self.y_train - self.prediction))*1000000,2)
        self.average_absolute_error = round((np.mean(np.abs(self.y_train - self.prediction)))*1000000,2)
        self.maximum_error = round(np.max((np.abs(self.y_train - self.prediction)) * 1000000),2)
        self.NN_Time_Taken = round(float(self.stop - self.start),2)
        self.Average_Time_Taken = round((self.NN_Time_Taken/self.x_train.shape[0])*1000000,2)
        self.std_dev = round((np.std(self.y_train - self.prediction)) * 1000000,2)
        greater_than_one_percent = np.sum((self.prediction - self.y_train) > 0.01)
        percentage_greater_than_one_percent = (np.sum((self.prediction - self.y_train) > 0.01)/self.x_train.shape[0])*100.0
        greater_than_half_percent = np.sum((self.prediction - self.y_train) > 0.005)
        percentage_greater_than_half_percent = (np.sum((self.prediction - self.y_train) > 0.005)/self.x_train.shape[0])*100.0
        greater_than_quarter_percent = np.sum((self.prediction - self.y_train) > 0.0025)
        percentage_greater_than_quarter_percent = (np.sum((self.prediction - self.y_train) > 0.0025)/self.x_train.shape[0])*100.0
        lesser_than_one_percent = np.sum((self.prediction - self.y_train) < -0.01)
        percentage_lesser_than_one_percent = (np.sum((self.prediction - self.y_train) < -0.01)/self.x_train.shape[0])*100.0
        lesser_than_half_percent = np.sum((self.prediction - self.y_train) < -0.005)
        percentage_lesser_than_half_percent = (np.sum((self.prediction - self.y_train) < -0.005)/self.x_train.shape[0])*100.0
        lesser_than_quarter_percent = np.sum((self.prediction - self.y_train) < -0.0025)
        percentage_lesser_than_quarter_percent = (np.sum((self.prediction - self.y_train) < -0.0025)/self.x_train.shape[0])*100.0
        st.write("For the testing dataset of ",self.x_train.shape[0],
                 " Options, we look at the statistics of the error distribution. Therefore an error of $10,000 would correspond to a 1% error" )
        st.markdown("The error values are quoted on a notional of a million dollars")
        st.write("Average Absolute error: $", self.average_absolute_error, " per million")
        st.write("Average error: $",self.average_error," per million")
        st.write("Maximum error: $", self.maximum_error," per million")
        st.write("Total Time Taken to price ",self.x_train.shape[0]," options is ",self.NN_Time_Taken," seconds")
        st.write("Time Taken is ", self.Average_Time_Taken,"microseconds per option")
        st.write("Percentage of Options with error > +/-1% : " + "{:.5%}".format( (greater_than_one_percent+lesser_than_one_percent)/ self.x_train.shape[0]))
        st.write("Percentage of Options with error > +/-0.5% : " + "{:.5%}".format((greater_than_half_percent + lesser_than_half_percent) / self.x_train.shape[0]))
        st.write("Percentage of Options with error > +/-0.25% : " + "{:.5%}".format((greater_than_quarter_percent + lesser_than_quarter_percent) / self.x_train.shape[0]))
        st.write("The plot below is a distribution of the error values along with vertical lines marking the 1%, 0.5% and 0.25% error values along with "
                 "the number and percentage of total options which have error values that exceed these bounds. Please enlarge the chart by clicking the enlarge option on the top right corner of the chart")
        a4_dims = (20, 15)
        fig, ax = plt.subplots(figsize=a4_dims)
        error = pd.DataFrame((self.prediction - self.y_train)) * 100.0
        error.columns = ['Error']
        error_plot = sns.kdeplot(ax=ax, data=error['Error'], shade=True)
        p = plt.axvline(-0.25, 0, 20, color="red")
        p = plt.axvline(0.25, 0, 20, color="red")
        p = plt.axvline(-0.5, 0, 20, color="green")
        p = plt.axvline(0.5, 0, 20, color="green")
        p = plt.axvline(-1, 0, 20, color="blue")
        p = plt.axvline(1, 0, 20, color="blue")
        p = plt.text(-1.0 + 0.02, 10.0, str(lesser_than_one_percent), fontsize=16)
        p = plt.text(-1.0 + 0.02, 9.0, str(round(percentage_lesser_than_one_percent,3)), fontsize=16)
        p = plt.text(-0.5, 10.0, str(lesser_than_half_percent), fontsize=16)
        p = plt.text(-0.5, 9.0, str(round(percentage_lesser_than_half_percent,3)), fontsize=16)
        p = plt.text(-0.25, 10.0, str(lesser_than_quarter_percent), fontsize=16)
        p = plt.text(-0.25, 9.0, str(round(percentage_lesser_than_quarter_percent,3)), fontsize=16)
        p = plt.text(0.25 - 0.11, 10.0, str(greater_than_quarter_percent), fontsize=16)
        p = plt.text(0.25 - 0.12, 9.0, str(round(percentage_greater_than_quarter_percent,3)), fontsize=16)
        p = plt.text(0.5 - 0.07, 10.0, str(greater_than_half_percent), fontsize=16)
        p = plt.text(0.5 - 0.09, 9.0, str(round(percentage_greater_than_half_percent,3)), fontsize=16)
        p = plt.text(1.0 - 0.05, 10.0, str(greater_than_one_percent), fontsize=16)
        p = plt.text(1.0 - 0.05, 9.0, str(round(percentage_greater_than_one_percent,3)), fontsize=16)
        st.pyplot()

class European_Call_Option(Derivative):

    def __init__(self):
        Derivative.__init__(self)
        Discount_factor = np.exp(-self.Risk_free_Rate * self.Time_to_Maturity)
        Dividend_rate=0.0
        #Dividend_rate = st.number_input("Enter the Dividend rate rate: ", key="q", value=0.05, min_value=0.0,max_value=0.1)
        Dividend_factor = np.exp(-Dividend_rate * self.Time_to_Maturity)
        self.Number_of_Paths = st.number_input("Enter thenumber of paths for Simulation: ", key="Paths", value=25000,min_value=2000, max_value=100000)
        d1, d2 = self.calc_d1_d2(self.Stock_Price, self.Strike_Price, self.Time_to_Maturity, self.Volatility, self.Risk_free_Rate,Dividend_rate)
        self.x_train = np.array([self.Stock_Price, self.Strike_Price, self.Time_to_Maturity, self.Volatility, self.Risk_free_Rate, Discount_factor, Dividend_rate,Dividend_factor, d1, d2]).reshape(1, -1)
        #st.write(self.x_train)

    def calc_d1_d2(self,S, K, T, sigma, r, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return (d1, d2)

    def payoff(self,ST, strike, r, T):
        payoff = np.exp(-r * T) * np.maximum(ST - strike , 0)
        return (payoff)

    def Monte_Carlo(self):
        n=self.Number_of_Paths
        Numer_of_Periods=int(self.Time_to_Maturity * 100)+1
        start = time.time()
        self.Simulated_Price_Paths(self.Stock_Price, self.Strike_Price,  self.Risk_free_Rate, self.Volatility, self.Time_to_Maturity, n, Numer_of_Periods)
        Final_Payoff = self.payoff(self.Simulated_Prices[Numer_of_Periods - 1, :],self.Strike_Price,self.Risk_free_Rate,self.Time_to_Maturity)
        Final_Payoff_Antithetic = self.payoff(self.Simulated_Prices_Antithetic[Numer_of_Periods - 1, :],self.Strike_Price,self.Risk_free_Rate,self.Time_to_Maturity)
        alpha = -(np.corrcoef(self.Simulated_Prices[Numer_of_Periods - 1, :], Final_Payoff)[0, 1]) * (np.std(Final_Payoff) / np.std(self.Simulated_Prices[Numer_of_Periods - 1, :]))
        Final_Payoff = Final_Payoff + alpha * (self.Simulated_Prices[Numer_of_Periods - 1, :] - self.Stock_Price * np.exp(self.Risk_free_Rate * self.Time_to_Maturity))
        alpha = -(np.corrcoef(self.Simulated_Prices_Antithetic[Numer_of_Periods - 1, :], Final_Payoff_Antithetic)[0, 1]) * (np.std(Final_Payoff_Antithetic) / np.std(self.Simulated_Prices_Antithetic[Numer_of_Periods - 1, :]))
        Final_Payoff_Antithetic = Final_Payoff_Antithetic + alpha * (self.Simulated_Prices_Antithetic[Numer_of_Periods - 1, :] - self.Stock_Price * np.exp(self.Risk_free_Rate * self.Time_to_Maturity))
        Final_Payoff = np.mean([Final_Payoff, Final_Payoff_Antithetic], axis=0)
        stop = time.time()
        self.MC_Option_Price = round(float(np.mean(Final_Payoff)), 6)
        self.MC_Time_Taken = round(float(stop- start) * 1000, 0)
        self.MC_Stderr = int(round((np.std(Final_Payoff)/np.sqrt(n)), 8) * 200000)

class American_Put_Option(Derivative):

    def __init__(self):
        Derivative.__init__(self)
        Discount_factor = np.exp(-self.Risk_free_Rate * self.Time_to_Maturity)
        self.Number_of_Paths = st.number_input("Enter thenumber of paths for Simulation: ", key="Paths", value=50000,min_value=2000, max_value=100000)
        d1, d2, self.BS_Option_Price, Intrinsic_Value = self.calc_put_option(self.Stock_Price, self.Time_to_Maturity,self.Volatility, self.Risk_free_Rate)
        self.x_train = np.array([self.Stock_Price, self.Time_to_Maturity, self.Volatility, self.Risk_free_Rate, d1, d2,Discount_factor,self.BS_Option_Price,Intrinsic_Value]).reshape(1, -1)

    def calc_put_option(self,S, T, sigma, r):
        d1 = (np.log(S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        Option_Price = (np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        Intrinsic_Value = np.maximum(np.exp(-r * T) - S,0)
        return (-d1, -d2, Option_Price, Intrinsic_Value)

    def payoff(self, ST, strike, r, T):
        payoff = np.exp(-r * T) * np.maximum(strike - ST , 0)
        return (payoff)

    def Monte_Carlo(self):
        n=self.Number_of_Paths
        N=int(self.Time_to_Maturity * 100)+1
        start = time.time()
        self.Simulated_Price_Paths(self.Stock_Price, self.Strike_Price, self.Risk_free_Rate, self.Volatility,self.Time_to_Maturity, n, N)
        self.Simulated_Prices=np.transpose(self.Simulated_Prices)
        self.Simulated_Prices_Antithetic = np.transpose(self.Simulated_Prices_Antithetic)
        self.Simulated_Prices = np.concatenate((np.ones((n, 1)) * self.Stock_Price, self.Simulated_Prices), axis=1)
        self.Simulated_Prices_Antithetic = np.concatenate((np.ones((n, 1)) * self.Stock_Price, self.Simulated_Prices_Antithetic), axis=1)
        Final_Prices = self.Simulated_Prices[:, N - 1]
        Final_Prices_Antithetic = self.Simulated_Prices_Antithetic[:, N - 1]
        Average_Payoff = self.payoff(Final_Prices, self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        Average_Payoff_Antithetic = self.payoff(Final_Prices_Antithetic, self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        S = np.concatenate((self.Simulated_Prices, self.Simulated_Prices_Antithetic), axis=0)
        time_points = np.linspace(0, self.Time_to_Maturity, num=N + 1, endpoint=True)
        Cash_Flow = np.zeros((2 * n, N + 1))
        Cash_Flow[:, N] = np.maximum(0, self.Strike_Price - S[:, N])
        for j in range(N - 1, -1, -1):
            payoff_vector = np.maximum(0, self.Strike_Price - S[:, j])
            Y = np.sum(np.exp(-self.Risk_free_Rate * (time_points[j + 1:] - time_points[j])) * Cash_Flow[:, j + 1:], axis=1)
            continuation_vector = self.Regression_Function(S[:, j], self.Strike_Price, Y)
            Cash_Flow[:, j] = (payoff_vector >= np.array(continuation_vector)) * payoff_vector
            Cash_Flow[Cash_Flow[:, j] > 0, j + 1:] = 0
        Cash_Flow_1 = Cash_Flow[0:n:1, :]
        Cash_Flow_2 = Cash_Flow[n:2 * n:1, :]
        Cash_Flow_1 = np.exp(-self.Risk_free_Rate * time_points) * Cash_Flow_1
        Cash_Flow_2 = np.exp(-self.Risk_free_Rate * time_points) * Cash_Flow_2
        price1 = (np.sum((Cash_Flow_1), axis=1))
        price2 = (np.sum((Cash_Flow_2), axis=1))
        Corr_coef = np.corrcoef(price1, Average_Payoff)[0, 1]
        if (np.std(Average_Payoff) == 0.0):
            alpha = 0.0
        else:
            alpha = -Corr_coef * (np.std(price1) / np.std(Average_Payoff))
        price1 = price1 + alpha * (Average_Payoff - self.BS_Option_Price)
        Corr_coef = np.corrcoef(price2, Average_Payoff_Antithetic)[0, 1]
        if (np.std(Average_Payoff_Antithetic) == 0.0):
            alpha = 0.0
        else:
            alpha = -Corr_coef * (np.std(price1) / np.std(Average_Payoff_Antithetic))
        price2 = price2 + alpha * (Average_Payoff_Antithetic - self.BS_Option_Price)
        price = (price2 + price1) / 2
        stop = time.time()
        self.MC_Option_Price = round(float(np.mean(price)), 6)
        self.MC_Time_Taken = round(float(stop - start) * 1000, 0)
        self.MC_Stderr = int(round((np.std(price) / np.sqrt(n)), 8) * 200000)

    def Regression_Function(self,St, K, DCF):
        L0 = np.exp(-St / (K * 2.0))
        L1 = np.exp(-St / (K * 2.0)) * (1.0 - (St / K))
        L2 = np.exp(-St / (K * 2.0)) * (1.0 - 2.0 * (St / K) + ((St / K) ** 2 / 2.0))
        A = np.vstack([np.ones(len(St)), L0, L1, L2]).T
        A[np.where(St > K), :] = 0.0
        B0, B1, B2, B3 = np.linalg.lstsq(A, DCF, rcond=None)[0]
        Continuation_Value = B0 + B1 * L0 + B2 * L1 + B3 * L2
        Continuation_Value[np.where(St > K)] = 0
        return Continuation_Value

class Asian_Call_Option(Derivative):

    def __init__(self):
        Derivative.__init__(self)
        self.Number_of_Paths = st.number_input("Enter thenumber of paths for Simulation: ", key="Paths", value=5000,min_value=2000, max_value=100000)
        self.Geometric_Option = self.Black_Scholes_Geometric_Exact(self.Stock_Price, self.Strike_Price, self.Risk_free_Rate, self.Volatility,self.Time_to_Maturity, self.Time_to_Maturity*100)
        self.x_train = np.array([self.Stock_Price, self.Time_to_Maturity, self.Volatility, self.Risk_free_Rate,self.Geometric_Option]).reshape(1, -1)

    def Black_Scholes_Geometric_Exact(self,S, K, r, sigma1, T, n):
        sigma = sigma1 * np.sqrt((n + 1.0) * (2.0 * n + 1.0) / (6.0 * n ** 2))
        q = (r * (n - 1.0) / (2.0 * n)) + (sigma1 ** 2.0) * ((n + 1.0) * (n - 1.0) / (12.0 * n ** 2))
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        call = (np.exp(-q * T) * S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        return call

    def payoff(self, ST, strike, r, T):
        payoff = np.exp(-r * T) * np.maximum(ST -strike, 0)
        return (payoff)

    def Monte_Carlo(self):
        n = self.Number_of_Paths
        N = int(self.Time_to_Maturity * 100)+1
        start = time.time()
        self.Simulated_Price_Paths(self.Stock_Price, self.Strike_Price, self.Risk_free_Rate, self.Volatility,self.Time_to_Maturity, n, N)
        Average_Payoff = self.payoff( self.Simulated_Prices.mean(0), self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        Average_Payoff_A = self.payoff(self.Simulated_Prices_Antithetic.mean(0), self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        Geometric_mean = stats.gmean( self.Simulated_Prices, axis=0)
        Geometric_Option_Prices = self.payoff(Geometric_mean, self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        Geometric_mean_A = stats.gmean(self.Simulated_Prices_Antithetic, axis=0)
        Geometric_Option_Prices_A = self.payoff(Geometric_mean_A, self.Strike_Price, self.Risk_free_Rate, self.Time_to_Maturity)
        if (np.std(Geometric_Option_Prices) == 0.0):
            alpha = 0.0
        else:
            alpha = -(np.corrcoef(Geometric_Option_Prices, Average_Payoff)[0, 1]) * (np.std(Average_Payoff) / np.std(Geometric_Option_Prices))
        Average_Payoff = Average_Payoff + alpha * (Geometric_Option_Prices - self.Geometric_Option)
        if (np.std(Geometric_Option_Prices_A) == 0.0):
            alpha_A = 0.0
        else:
            alpha_A = -(np.corrcoef(Geometric_Option_Prices_A, Average_Payoff_A)[0, 1]) * (np.std(Average_Payoff_A) / np.std(Geometric_Option_Prices_A))
        Average_Payoff_A = Average_Payoff_A + alpha_A * (Geometric_Option_Prices_A - self.Geometric_Option)
        Average_Payoff = np.mean([Average_Payoff, Average_Payoff_A], axis=0)
        stop=time.time()
        self.MC_Option_Price = round(float(np.mean(Average_Payoff)), 6)
        self.MC_Time_Taken = round(float(stop - start) * 1000, 0)
        self.MC_Stderr = int(round((np.std(Average_Payoff) / np.sqrt(n)), 8) * 200000)

value = st.sidebar.selectbox("Select Type of Option", ["European Call Option","American Put Option", "Asian Call Option"])

if (value == "European Call Option"):
    st.title("European Call Option")
    European_Call_Option=European_Call_Option()
    st.markdown("To test the pricing for a single option, enter the paramters in the fields shown above and click the button")
    if st.button('Price European Call Option'):
        European_Call_Option.Monte_Carlo()
        European_Call_Option.Compile_Neural_Network('European_Option_Final_Weights_mse.hdf5')
        European_Call_Option.Predict_Neural_Network()
        European_Call_Option.print_results()
    st.write("")
    st.write("")
    st.markdown("To Perform a more comprehensive test of the Neural Network Pricing we will price a dataset consisting of around a million Options")
    st.markdown("This test dataset is built by as broad of a coverage over the input variables as possible")
    if st.button('Perform Speed & Accuracy test'):
        European_Call_Option.Compile_Neural_Network('European_Option_Final_Weights_mse.hdf5')
        European_Call_Option.speed_accuracy_test('European_option_x_train_subset.h5','European_option_y_train_subset.h5')

elif (value == "American Put Option"):
    st.title("American Put Option")
    American_Put_Option=American_Put_Option()
    st.markdown("To test the pricing for a single option, enter the paramters in the fields shown above and click the button")
    if st.button('Price American Put Option'):
        American_Put_Option.Monte_Carlo()
        American_Put_Option.Compile_Neural_Network('American_Option_Final_weights_mse.hdf5')
        American_Put_Option.Predict_Neural_Network()
        American_Put_Option.print_results()
    st.write("")
    st.write("")
    st.markdown("To Perform a more comprehensive test of the Neural Network Pricing we will price a dataset consisting of around a million Options")
    st.markdown("This test dataset is built by as broad of a coverage over the input variables as possible")
    if st.button('Perform Speed & Accuracy test'):
        American_Put_Option.Compile_Neural_Network('American_Option_Final_weights_mse.hdf5')
        American_Put_Option.speed_accuracy_test('American_option_x_train_subset.h5','American_option_y_train_subset.h5')
else:
    st.title("Asian Call Option")
    Asian_Call_Option = Asian_Call_Option()
    st.markdown("To test the pricing for a single option, enter the paramters in the fields shown above and click the button")
    if st.button('Price Asian Call Option'):
        Asian_Call_Option.Monte_Carlo()
        Asian_Call_Option.Compile_Neural_Network('Asian_Option_Final_weights_mse.hdf5')
        Asian_Call_Option.Predict_Neural_Network()
        Asian_Call_Option.print_results()
    st.write("")
    st.write("")
    st.markdown("To Perform a more comprehensive test of the Neural Network Pricing we will price a dataset consisting of around a million Options")
    st.markdown("This test dataset is built by as broad of a coverage over the input variables as possible")
    if st.button('Perform Speed & Accuracy test'):
        Asian_Call_Option.Compile_Neural_Network('Asian_Option_Final_weights_mse.hdf5')
        Asian_Call_Option.speed_accuracy_test('Asian_option_x_train_subset.h5','Asian_option_y_train_subset.h5')