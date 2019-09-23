from selenium import webdriver

browser = webdriver.Safari()
browser.get('http://localhost:8000')

assert browser.page_source.find('install')
