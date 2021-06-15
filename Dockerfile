FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python3", "./main.py" , "--input", "facebook/facebook.gpickle", \
                              "--output", "facebook/facebook_deepwalk_custom.embedding ", \
                              "--results", "facebook/facebook_deepwalk_custom_logisticalregression.csv", \
                              "--method", "deepwalk_custom", \
                              "--classifier", "logisticalregression", \
                              "--evaluation", "node-classification", \
                              "--embed", "true", "--node-ml-target", "ml_target"]
