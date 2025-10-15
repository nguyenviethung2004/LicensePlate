FROM python:3.9.11-slim

# Set biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Thư mục làm việc trong container
WORKDIR /app

# Copy file requirements vào container
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở port Flask
EXPOSE 5000

# Chạy Flask API
CMD ["python", "src/app.py"]