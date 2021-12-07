FROM python:3.9.9

WORKDIR /Pyspark_ML

COPY . /Pyspark_ML

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get install -y ant && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    rm -rf /var/cache/oracle-jdk8-installer;

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
RUN export JAVA_HOME


RUN pip install pandas
RUN pip install pyspark
RUN pip install numpy
RUN pip install imblearn
RUN pip install scikit-learn
RUN pip install py4j
RUN pip install scipy

CMD ["python3", "./main.py"]