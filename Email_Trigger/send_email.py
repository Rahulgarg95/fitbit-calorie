import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import socket

class email:
    def __init__(self):
        self.user='rahulpython95@gmail.com'
        self.passwd='Qwer!234'

    def trigger_mail(self,to_addr,cc_addr,msg):
        try:
            receivers=to_addr
            receivers.extend(cc_addr)
            msg['To'] = ", ".join(to_addr)
            from_addr='FitBit Calories <fitbitalert@gmail.com>'
            msg['From'] = from_addr
            if len(cc_addr)>0:
                msg['Cc'] = ", ".join(cc_addr)
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login(self.user, self.passwd)
            s.sendmail(from_addr, receivers, msg.as_string())
            s.close()
        except Exception as e:
            raise Exception(e)