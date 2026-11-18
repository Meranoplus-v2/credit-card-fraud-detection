FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy requirements first (so docker caches this layer)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy everything else
COPY main.py .
COPY feedback.py .
COPY model.pkl .
COPY scaler.pkl .

# expose the port
EXPOSE 8000

# run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]