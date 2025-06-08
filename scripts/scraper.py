from google_play_scraper import Sort, reviews
import csv
from datetime import datetime
import time
import logging
# import schedule

logging.basicConfig(filename='scrapper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bank_name = "BOA"
def scrape_play_store_reviews():
    # APP_ID = 'com.dashen.dashensuperapp'
    # APP_ID = 'com.combanketh.mobilebanking'
    APP_ID = 'com.boa.boaMobileBanking'

    logging.info("Fetching reviews")
    try:
        results,_=reviews(
            APP_ID,
            lang='en',
            country='us',
            sort = Sort.NEWEST,
            count = 1000,
            filter_score_with=None
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/{bank_name}_{timestamp}.csv'

        with open(filename,mode='w', newline='',encoding='utf-8') as file:
            writer = csv.DictWriter(file,fieldnames=['review_text','rating','date','bank_name','source'])
            writer.writeheader()
            for entry in results:
                writer.writerow({
                    'review_text':entry['content'],
                    'rating':entry['score'],
                    'date':entry['at'].strftime('%Y-%m-%d'),
                    'bank_name':bank_name,
                    'source':'Google Play',
                })
        logging.info(f'Saved{len(results)} reviews to {filename}')
    except Exception as e:
        logging.error(f"Error occurred: {e}")

scrape_play_store_reviews()
# schedule.every().day.at("01:00").do(scrape_play_store_reviews)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

