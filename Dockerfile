FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python3", "./main.py" , "--input", "facebook_pages.edgelist", \
                              "--output", "facebook_pages.txt", \
                              "--results", "facebook_pages.csv", \
                              "--method", "deepwalk_custom", \
                              "--classifier", "logisticalregression", \
                              "--evaluation", "node-classification"]