import feedparser
from flask import Flask, url_for
from flask import render_template
from flask import request
import numpy as np
import time
import re

app = Flask(__name__)

RSS_FEEDS = {'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
             'techcrunch': 'http://feeds.feedburner.com/TechCrunch/startups'
			}

# following a file - See https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file
def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line
        
def get_summary(story):
    
    inference_in_file = "inference.in"
    inference_out_file = "inference.out"
    
    # wait on inference outfile for output
    inference_file = open(inference_out_file,"r")
    summaries = follow(inference_file)
    
    # WRITE OUT to inference in file
    transaction_id = np.random.randint(0,10000)
    with open(inference_in_file,"a+") as in_file:
        in_file.write(str(transaction_id) + "," + story + "\n")

    # wait for the summary to show up - check for corrresponding transaction id
    summary = ""
    for line in summaries:
        
        print("Summary line:",line)

        # format of line is transaction_id,test_story
        trans_id,summary = line.split(",")
        
        print("Splitting summary line. trans_id:%d,transaction_id:%d,summary:%s" % (int(trans_id),transaction_id,summary))
        
        if int(trans_id) == transaction_id:
            break
           
    print("Returning summary:",summary)
    
    return summary

@app.route("/")
def get_news():

        background_image = "blank-business-close-up-1007025.jpg"
        background_url = url_for('static', filename='blank-business-close-up-1007025.jpg')
        
        story = request.args.get("story")
        
        summary = "Nothing here yet...please enter and submit some text and then something should show up..."
        if story != None:
            print("Processing story:",story)
            
            # clean up story - keep only words
            WORD_RE = re.compile(r"[\w]+")
            clean_story = " ".join(WORD_RE.findall(story))
            
            print("Cleaned story:",clean_story)
            
            summary = get_summary(clean_story)
            
            print("Summary returned from inference module:",summary)
        
        return render_template("home_form.html",summary=summary,story=story,background_url=background_url)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=80, debug=True)
