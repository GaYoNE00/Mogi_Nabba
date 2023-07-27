from flask import Flask, request
import mysql.connector
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re

#뷰티풀 숩
url = "https://www.weather.go.kr/home/main-now-weather.jsp?gubun=2&myPointCode=&unit=M"



# MySQL 연결 설정
db = mysql.connector.connect(
    host="192.168.0.49",
    user="mogi_user",
    password="2213",
    database="mogi"
)

# Flask 앱 인스턴스 생성
app = Flask(__name__)

@app.route('/')
def home():
    return 'Mogi Home!'

# URL에 파라미터 값에 따라 MySQL에 INSERT
@app.route('/insert', methods=['GET'])
def insert_data():
    # URL에서 파라미터 값 추출
    temp = request.args.get('temp')

    # MySQL 커서 생성
    cursor = db.cursor()

    # 현재 날짜와 시간 가져오기
    now = datetime.now()
    date = now.date()  # 현재 날짜
    time = now.time()  # 현재 시간
    # 웹 페이지에 GET 요청을 보냄
    response = requests.get(url)

    # 요청에 성공한 경우 HTML 코드를 가져옴
    if response.status_code == 200:
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        raindiv = soup.find(class_='rainfall')
        if raindiv:
            po_seoul = raindiv.find(class_='po_seoul')
            if po_seoul:
                rainfall = po_seoul.find(class_='rainfall')
                if rainfall:
                    rain = rainfall.text
                    numbers_only = re.findall(r'\d+\.\d+|\d+', rain)
                    rain = float(numbers_only[0])
                else:
                    print("비가 오지 않습니다.")
                    rain = 0.0

            else:
                rain = 0.0
                print("서울 데이터를 찾을 수 없습니다.")
    else:
        print("페이지를 가져오는데 실패했습니다.")

    # INSERT 쿼리 실행
    query = "INSERT INTO test (temp, date, time, rain) VALUES (%s, %s, %s, %s)"
    values = (temp, date, time, rain,)
    cursor.execute(query, values)

    # 데이터베이스 변경 사항 커밋
    db.commit()

    # 커넥션 및 커서 닫기
    cursor.close()
    print(rain)
    return 'Data inserted successfully.'

@app.route('/data')
def data():
    # MySQL 커서 생성
    cursor = db.cursor()

    # 뷰 조회 쿼리 실행
    query = "SELECT * FROM daily_view"
    cursor.execute(query)

    # 쿼리 결과 가져오기
    result = cursor.fetchall()

    # 결과를 문자열로 변환하여 반환
    data_str = ""
    for row in result:
        date = row[0]
        max_temp = row[1]
        min_temp = row[2]
        avg_temp = row[3]
        data_str += f"Date: {date}, Max Temp: {max_temp}, Min Temp: {min_temp}, Avg Temp: {avg_temp}<br>"

    # 커넥션 및 커서 닫기
    cursor.close()

    return data_str



# 앱 실행
if __name__ == '__main__':
    app.run(host='192.168.0.32')