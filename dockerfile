# Start

FROM python:3.12-slim

WORKDIR /thesis-app

RUN apt-get update && apt-get install -y git curl

RUN git clone https://github.com/sahacha/thesis-app.git

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
