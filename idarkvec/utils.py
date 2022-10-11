from datetime import datetime, timedelta

def get_next_day(start):
    start = datetime.strptime(start, '%Y%m%d')
    day = start+timedelta(days=1)
    day = day.strftime('%Y%m%d')
    
    return day

def get_prev_day(start):
    start = datetime.strptime(start, '%Y%m%d')
    day = start-timedelta(days=1)
    day = day.strftime('%Y%m%d')
    
    return day