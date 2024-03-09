# here we doing web application where we having form , 
# here we give all our input data that is required to presict students performances 
# here we considering using flask app 


from flask import Flask , request , render_template  
import numpy as np 
import pandas as np 

from sklearn.preprocessing import StandardScaler   
from src.pipeline.predict_pipeline import CustomData , predictpipeline 

application=Flask(__name__)
app=application 

@app.route('/') 
def index() :
    return render_template('index.html')


@app.route('/predictdata' , methods=['GET','POST']) 
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  
    else :                                                                      # in the post part we have to capture data , we have to do standarscaling then we do prediction
        data=CustomData(
            gender=request.form.get('gender'),                                    # here we add(get) all information coming from webrl as input 
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        
        pred_df=data.get_data_as_data_frame()                # we are converted our input data in to dataframe 
        print(pred_df)  
        print("before prediction")             
         
        predict_pipeline= predictpipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)                       # throght predict opreation we get our output(here we are using all power of of predict function from pipeline.py)  
        print("after Prediction")
        return render_template('home.html',results=results[0])         # our putput it is in the list format 
        
        
        
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        
        
