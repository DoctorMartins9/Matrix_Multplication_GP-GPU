import csv
import sys


# Open csv
f = open(str(sys.argv[1]))
csv_f = csv.reader(f)

# From csv to list
vdd_in = []
vdd_volt = []
gpu_in = []
gpu_volt = []

for row in csv_f:
	vdd_in.append(row[9])
	vdd_volt.append(row[8])
	gpu_in.append(row[2])
	gpu_volt.append(row[3])

# delete header
del vdd_in[0]
del vdd_volt[0]
del gpu_in[0]
del gpu_volt[0]

# From string to int & power array
count = 0
vdd_power = []
gpu_power = []
while count < len(vdd_in):
	vdd_in[count] = float(vdd_in[count])
	vdd_volt[count] = float(vdd_volt[count])
	
	gpu_in[count] = float(gpu_in[count])
	gpu_volt[count] = float(gpu_volt[count])
	
	vdd_power.append(vdd_in[count] * vdd_volt[count])	
	gpu_power.append(gpu_in[count] * gpu_volt[count])

	count +=1

# Results
result = []

vdd_power = sorted(vdd_power)
result.append(str(sys.argv[1]))
result.append(sum(vdd_power) / len(vdd_power))
result.append(vdd_power[len(vdd_power)-1])

gpu_power = sorted(gpu_power)
result.append(sum(gpu_power)/len(gpu_power))
result.append(gpu_power[len(gpu_power)-1])

print "-----------------------------------------"
print result[0] 
print "avg power:     " + str(result[1]/1000000)
print "max power:     " + str(result[2]/1000000)
print "gpu avg power: " + str(result[3]/1000000)
print "gpu max power: " + str(result[4]/1000000)
print "-----------------------------------------"