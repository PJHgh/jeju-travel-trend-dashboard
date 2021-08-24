# Travel Trends in Instagram
> ***os 환경 : window 10 pro***
---
## Set up
1. Install Requirements
    ```dotnetcli
    pip install -r requirements.txt
    ```
2. Instargram Crawling & Scrapping
    - Chrome version check
        > chrome 실행 후 chrome://settings/help 접속하여 version 확인
    - ChromeDriver Download
        > https://chromedriver.chromium.org/downloads 접속 후 자신의 chrome version과 맞는 것을 찾아 download
    - 해당 경로에 위치
        ```dotnetcli
        .\jeju-travel-trend-dashboard\etc\chromedriver.exe
        ```
    - Enter email & password
        - insta_crawling.py
            ```python
            def main(keyword_list):
                email = '*********'
                password = '*********'
            
                insta = InstaKeywordCrawling(email, passwokrd)
                insta.login()
                for keyword in list(keyword_list.split(',')):
                    insta.new_crawling(keyword)
                    insta.save_result()
                insta.driver.quit()
            ```
    - Run script
        ```dotnetcli
        bash ./crawling.sh
        ```

3. kakao API setting
    > 1. kakao API를 사용하기 위해 개발자 등록을 해야합니다.
    > 2. utils.py 의 get_lat_and_lon 함수안에서 headers의 Authorization의 value를 수정해야 합니다.
    - Get API key
        > 1. https://developers.kakao.com/ 접속
        > 2. 로그인 후 개발자 등록, 앱 생성
        > 3. REST API Key 복사
    - Enter REST API Key
        - utils.py
            ```python
            def get_lat_and_lon(place):
                insta_places_info = pd.read_csv('./etc/insta_places_info.csv')
                insta_places_info['insta_names'] = insta_places_info['insta_names'].apply(lambda x:list(map(lambda x:re.sub(r"[']", "", x).strip(), x[1:-1].split(','))))
                searched_places = set(insta_places_info['insta_names'].sum())
            
                if place.strip() in searched_places:
                    name, lat, lon = insta_places_info[insta_places_info['insta_names'].apply(lambda x:place.strip() in x)][['name', 'lat', 'lon']].iloc[0]
                    return [name, lat, lon, place.strip()]
            
                url = f'https://dapi.kakao.com/v2/local/search/keyword.json?query=제주도 {place}'
                headers = {"Authorization":"******"}
            
                places = requests.get(url, headers=headers).json()['documents']
                place_loc = places[0]
                name = place_loc['place_name']
                lon, lat = place_loc['x'], place_loc['y']
                
                return [name, lat, lon, place]
            ```
4. Elasticsearch setting
    - Elasticsearch download
        1. https://www.elastic.co/downloads/elasticsearch 접속
        2. window 압축 파일 다운로드 (version 확인)
        3. 압축 해제
    - directory setting
        ```
        jeju-travel-trend-dashboard
        ├── data
        │   └── all.csv
        ├── ***elasticsearch-"version"***
        ├── etc
        │   ├── chromedriver.exe
        │   ├── insta_places_info.csv
        │   └── stoptags
        ├── crawling.sh
        ├── insta_crawling.py
        ├── README.md
        ├── travel_trends.py
        ├── utils.py
        └── requirements.txt
        ```
    - Enter path
        - elasticsearch-"version"/config/elasticsearch.yml
            ```yaml
            # ----------------------------------- Paths ------------------------------------
            #
            # Path to directory where to store the data (separate multiple locations by comma):
            #
            path.data: C:\Users\user\Desktop\jeju-travel-trend-dashboard\elasticsearch-"version"\logs\data
            #
            # Path to log files:
            #
            path.logs: C:\Users\user\Desktop\jeju-travel-trend-dashboard\elasticsearch-"version"\logs
            #
            ```
    - Enter port
        - elasticsearch-"version"/config/elasticsearch.yml
            ```yaml
            # ---------------------------------- Network -----------------------------------
            #
            # By default Elasticsearch is only accessible on localhost. Set a different
            # address here to expose this node on the network:
            #
            #network.host: 192.168.0.1
            #
            # By default Elasticsearch listens for HTTP traffic on the first free port it
            # finds starting at 9200. Set a specific HTTP port here:
            #
            http.port: 9200
            #
            # For more information, consult the network module documentation.
            #
            ```
    - Run elasticsearch
        ```powershell
        ./elasticsearch-"version"/bin/elasticsearch.bat
        ```
1. Dashboard
    - Run Dashboard
        ```dotnetcli
        streamlit run travel_trends.py
        ```
    - address : http://localhost:8501/

        ![ezgif com-gif-maker](https://user-images.githubusercontent.com/51353039/130311572-e945b4a9-b476-4f27-b492-a238ec5f0295.gif)
    
