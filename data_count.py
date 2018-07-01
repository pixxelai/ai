import pandas as pd
import sys

if len(sys.argv) < 6:
	print("\nUsage: python data.py [State_Name] [Crop_Year-lb] [Crop_Year-ub] [Season] [Crop]")
	print("Example: python data.py Punjab 2001 2014 Kharif Cotton")
	print("Note: Use 'Cotton' for 'Cotton(lint)'\n")
	sys.exit()

if len(sys.argv) == 7:
	State_Name = sys.argv[1] + ' ' + sys.argv[2]
	Crop_Year_lb = int(sys.argv[3])
	Crop_Year_ub = int(sys.argv[4])
	Season = sys.argv[5]
	Crop = sys.argv[6]
else:
	State_Name = sys.argv[1]
	Crop_Year_lb = int(sys.argv[2])
	Crop_Year_ub = int(sys.argv[3])
	Season = sys.argv[4]
	Crop = sys.argv[5]

df=pd.read_csv('apy.csv')
df_query = df.copy()
df_query = df_query[df_query['State_Name'].str.contains(State_Name)]
df_query = df_query[df_query['Crop_Year']>=Crop_Year_lb]
df_query = df_query[df_query['Crop_Year']<=Crop_Year_ub]
df_query = df_query[df_query['Season'].str.contains(Season)]
df_query = df_query[df_query['Crop'].str.contains(Crop)]
output_string = ' '.join(sys.argv[1:])
print(output_string + ": %d" % df_query.shape[0])