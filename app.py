import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import csv
import pandas as pd
from transformers import pipeline
import io
import base64

import matplotlib
matplotlib.use('Agg')


pretrained = "mdhugol/indonesia-bert-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

sentiment_analysis = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer)

# labels = ['Negative', 'Neutral', 'Positive']

labels = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

# col_data = []
# count = 1


def predict(col_data, count):
    res = []
    for k in range(count):
        # input data prep
        input_sent = col_data[k]
        # input_sent = input_sent[:120]

        result = sentiment_analysis(input_sent)
        # print(result)     result=[{'label':,'score':}]
        status = labels[result[0]['label']]
        score = result[0]['score']
        #print(f'Text: {input_sent} | Label : {status} ({score * 100:.3f}%)')

        if (status == labels['LABEL_0']):
            res.append(1)
        elif (status == labels['LABEL_1']):
            res.append(0)
        else:
            res.append(-1)

    count_neutral = res.count(0)
    count_positive = res.count(1)
    count_negative = res.count(-1)

    # print("Count of Positive : ", count_positive, "\nCount of Neutral : ",
    #       count_neutral, "\nCount of Negative : ", count_negative)

    per_positive = (count_positive/count)*100
    per_netural = (count_neutral/count)*100
    per_negative = (count_negative/count)*100

    # print("Positive : ", per_positive, "\nNeutral : ",
    #       per_netural, "\nNegative : ", per_negative)
    print_Str = [per_positive, per_netural, per_negative, count]
    # Return the prediction result
    return print_Str


def create_pie_chart(data, filename):
    # Create the pie chart
    plt.pie(data, labels={"Positive", "Neutral",
            "Negative"}, autopct='%1.1f%%')
    plt.axis('equal')

    # Save the chart to a file
    plt.savefig(filename, format='png')

    # Encode the chart image to base64
    with open(filename, 'rb') as file:
        chart_image = base64.b64encode(file.read()).decode('utf-8')
    return chart_image


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    colName = request.form['string']
    csv_file = request.files['csv']

    # Save the CSV file temporarily and read the data
    csv_path = './uploads/temp.csv'
    csv_file.save(csv_path)
    data = pd.read_csv(csv_path)
    # data = pd.read_csv("./review_titanic.csv")
    # col_data = data['Reviews']
    col_data = data[colName]
    count = len(col_data)
    # print("Count of sentences which are given as input : ", count)
    result = predict(col_data, count)
    # print(result)
    filename = './static/pie_chart.png'
    chart_image = create_pie_chart(result[:3], filename)
    count = f'Number of reviews given in csv file : {count}'
    positive = f"Positive reviews percentage: {result[0]:.3f}%"
    neutral = f"Neutral reviews percentage: {result[1]:.3f}%"
    negative = f"Negative reviews percentage: {result[2]:.3f}%"
    return render_template('result.html', count=count, positive=positive, neutral=neutral, negative=negative)


if __name__ == '__main__':
    app.run(debug=True)
