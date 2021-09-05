import streamlit as st
import pandas as pd
import base64
import joblib
from random import randint
import streamlit.components.v1 as components
from prep import FeatureSelector, FeatureGemerator
import shap
shap.initjs()
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="answers.csv">Download answers.csv</a>'

def to_df(ID,Age,Exp,Incm,Fam,CCAvg,education,Mort,sa,cd,online,credit):
	eds = { 'Undergrad':0, 'Graduate':1,'Advanced/Professional':2}
	df = pd.DataFrame(index=[0])
	df['ID']=ID
	df['Age'] = Age
	df['Experience'] = Exp
	df['Income'] = Incm
	df['ZIP Code'] = randint(0,100)
	df['Family'] = Fam
	df['CCAvg'] = CCAvg
	df['Education'] = eds[education]
	df['Mortgage'] = Mort
	df['Securities Account'] = int(sa)
	df['CD Account'] = int(cd)
	df['Online'] = int(online)
	df['CreditCard'] = int(credit)
	return df
@st.cache
def gen_random_param():
	age = randint(21, 72)
	params = {
		'age' : age,
		'exp' : randint(0, age-18),
		'inc' : randint(5, 250),
		'fam' : randint(1, 4),
		'ccavg': randint(0, 10),
		'mort': randint(0, 350),
		'education': randint(0, 2),
		'sa': randint(0, 1),
		'cd': randint(0, 1),
		'online': randint(0, 1),
		'credit': randint(0, 1)
	}
	return params


model_pipeline = joblib.load('models/final_model.pkl')
target_name = ["won't accept",'will accept']
st.set_page_config(layout="wide")
st.title('Demo of personal loan prediction model')
st.subheader('For correct display, please use a __light theme__')
#st.write('<small>For the correct display of the results switch on to the light theme in the settings</small>', unsafe_allow_html=True)

with st.form('text'):
	random_person_button = st.form_submit_button('Generate random person')

	params = gen_random_param()
	if random_person_button:
		st.caching.clear_cache()
		params = gen_random_param()

	c1,c2,c3 = st.columns([1,1,1])
	c4,c5,c6 = st.columns([1,1,1])
	ID=randint(0,300)
	Age = c1.number_input("Enter client's age", min_value = 21, value = params['age'])
	Exp = c2.number_input("Enter client's professional experience", min_value = 0, value= params['exp'])
	Incm = c3.number_input("Enter client's annual income($000)", min_value=0, value = params['inc'], step = 5)
	Fam = c4.number_input("Enter the client's size family", min_value = 1, value = params['fam'])
	CCAvg = c5.number_input("Enter client's average spending on credit cards per month ($000)", min_value = 0.0, value = float(params['ccavg']))
	Mort = c6.number_input("Enter the mortgage size if any ($000)", min_value = 0, step = 50, value = params['mort'])
	education = st.selectbox("Select client's education level", ['Undergrad', 'Graduate','Advanced/Professional'], index = params['education'])
	sa = st.checkbox('Client has a securities account with the bank', value = params['sa'])
	cd = st.checkbox('Client has a certificate of deposit account with the bank', value = params['cd'])
	online = st.checkbox('Client use internet banking facilities', value = params['online'])
	credit = st.checkbox('Client use a credit card issued by the bank', value = params['credit'])
	button = st.form_submit_button('Click here to predict one person')
	if button:
		test_df = to_df(ID,Age,Exp,Incm,Fam,CCAvg,education,Mort,sa,cd,online,credit)
		#pipeline = joblib.load('data_engeneering_pipeline.pkl')
		ready_to_predict = model_pipeline[:-1].fit_transform(test_df)
		pred = model_pipeline.predict(test_df)
		prob = model_pipeline[-1].predict_proba(ready_to_predict)
		explainer = shap.TreeExplainer(model_pipeline.named_steps['cb_classifier'])
		shap_values = explainer.shap_values(ready_to_predict)
		name = f'Сlient will accept offer with {prob[0][1]:.3f} probability'
		if pred[0]==0:
			name+=f' (probability of __rejection__: {prob[0][0]:.3f})'
		labels =[
	    'Income($$$)',"Family size","ACCS","Education lvl", "Mortgage",
	    "securities account", "Certificate of deposit", "Internet banking facilities","Credit Card", "Has mortgage",
	    "Have family", "Graduated", "Mortgage is tercile","Income is tercile","ACCS is tercile"
	    ]

		st.write(name)
		
		st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:],
		 	ready_to_predict.iloc[0,:], out_names='probability of acceptance', feature_names=labels,link='logit'))
		st.write('<small>*ACCS — Average credit card spending</small>', unsafe_allow_html=True)
		st.markdown('Higher scores lead the model to predict 1 and lower scores lead the model to predict 0.'
		'With red representing features that pushed the model score higher, and blue representing features that pushed the score lower.'
		 'Features that had more of an impact on the score are located closer to the dividing boundary between red and blue, and the size of that impact is represented by the size of the bar.')





st.markdown('__OR__')

with st.form('table'):
	st.write('For the correct work, please use the table format below (with the right order and names of columns)')
	example_df = pd.DataFrame(index=['Data type'])
	example_df['ID']='int'
	example_df['Age'] = 'int'
	example_df['Experience'] = 'int'
	example_df['Income'] = 'int'
	example_df['ZIP Code'] = 'int'
	example_df['Family'] = 'int'
	example_df['CCAvg'] = 'float'
	example_df['Education'] =  'Undergrad: 0, Graduate: 1\nAdvanced/Professional:2'
	example_df['Mortgage'] = 'int or bool'
	example_df['Securities Account'] = 'int or bool'
	example_df['CD Account'] = 'int or bool'
	example_df['Online'] = 'int or bool'
	example_df['CreditCard'] = 'int or bool'
	example_df.index.name = 'Name of columns'
	st.table(example_df)
	uploaded_file = st.file_uploader("Upload a csv file", ["csv"])
	file_button = st.form_submit_button('Predict labels')



	if uploaded_file is not None:
		file = pd.read_csv(uploaded_file, sep= '[;,]', engine='python')


	if file_button:
		try:
			#pipeline = joblib.load('data_engeneering_pipeline.pkl')
			#ready_to_predict = pipeline.fit_transform(file)
			pred = model_pipeline.predict(file)
			file['Predictions'] = pred
			final_table_columns = ['ID', 'Predictions']
			file = file.drop(columns=[col for col in file if col not in final_table_columns])
			st.write(file)
			file['Predictions'] = file['Predictions'].astype('str')
			st.markdown(get_table_download_link(file), unsafe_allow_html=True)
		except:
			st.markdown('__Fail to predict labels__')
