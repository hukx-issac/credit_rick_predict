# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 12:42:02 2016

@author: Issac
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

def sendResultByEmail(email_content='',att_paths=None,from_who='sever',to_who='Issac'):
    # 第三方 SMTP 服务
    mail_host="smtp.cqu.edu.cn"  #设置服务器
    mail_user="20151313046@cqu.edu.cn"    #用户名
    mail_pass="hkx921023"   #口令 

    sender = '20151313046@cqu.edu.cn' 
    receivers = ['hkx418662942@qq.com']    # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
    
    #创建一个带附件的实例
    message = MIMEMultipart()
    message['From'] = Header(from_who, 'utf-8')
    message['To'] =  Header(to_who, 'utf-8')
    subject = 'Python 程序运行结束报告'     #主题
    message['Subject'] = Header(subject, 'utf-8') 
    
    #邮件正文内容
    message.attach(MIMEText(email_content, 'plain', 'utf-8'))
       
    if att_paths is not None:
        for att_path in att_paths:
            # 构造附件，传送当前目录下的 test.txt 文件
            att = MIMEText(open(att_path, 'rb').read(), 'base64', 'utf-8')
            att["Content-Type"] = 'application/octet-stream'
            # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
            basename = os.path.basename(att_path)
            att["Content-Disposition"] = 'attachment; filename="%s"' % basename
            message.attach(att)

    
    try:
        smtpObj = smtplib.SMTP() 
        smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user,mail_pass)  
        smtpObj.sendmail(sender, receivers, message.as_string())
        print "邮件发送成功"
    except smtplib.SMTPException:
        print "Error: 无法发送邮件"
        
if __name__ == '__main__':
    email_content = 'tesing !!!'
    from_who = 'my computer'
    sendResultByEmail(email_content,'output.csv')