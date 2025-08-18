# Step 1: Use official Python image
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app/Alpha1

# Step 3: Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy project files into container
COPY . .

# Step 5: Expose the port FastAPI runs on
EXPOSE 8000

# Step 6: Run FastAPI with uvicorn
CMD ["uvicorn", "apis:app", "--host", "0.0.0.0", "--port", "8000"]
