#
# Examples on how to use Sqlite3, II
# DB consultation part
#
import sqlite3

DB = './sqlite/cinema_db'

def main():

	# open a file called mydb with a SQLite3 DB
	db = sqlite3.connect(DB)

	# Get a cursor object
	cursor = db.cursor()

	# model execution
	cursor.execute('''SELECT city_id, population FROM cities''')

	#retrieve the first city
	# in this query, the order is undetermined, see ex. below
	one_city = cursor.fetchone()
	print((one_city[1]))

	all_rows = cursor.fetchall()
	for row in all_rows:
		print(('| {0} : {1}|'.format(row[0], row[1])))
	db.close()


	# Finish here: use SORT/ORDER BY to print the most popolous city

main()
