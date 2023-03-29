def count_glucose_windows(threshholdnumber: int):
    #print("Glucose level threshold number: {} | Meal threshold number: {}".format(threshholdnumber,mealcount))
    root="<root>Glucose window size: {} ".format(threshholdnumber)
    for filepaths in ALL_FILE_PATHS:
        data, patient_data = load(filepaths)
        stringpath = filepath_to_string(filepaths)
        for key in data.keys():
            data[key] = data[key].reset_index(drop=True)
        #print("{} Glucose level data amount: {} / {} ".format(stringpath,len(cleaned_data['glucose_level']),len(data['glucose_level'])))
        root+="<child>{} : Glucose level data amount: {} </child>".format(stringpath,count_continuous_glucose_windows(data['glucose_level'],threshholdnumber))
    root+="</root>"

    #tree.write("Patients.xml", encoding="utf-8", xml_declaration=True, method="xml",pretty_print=True)
    dom = minidom.parseString(root)
    pretty_xml_str = dom.toprettyxml()
    filename="Patients_windows{}.xml".format(threshholdnumber)
    with open(filename, "w") as f:
        f.write(pretty_xml_str)

def count_continuous_glucose_windows(data, thresholdglucose: int, time_step=pd.Timedelta(5, 'm')):
    counter=0
    count=0

    prev_ts = None
    for ts in data['ts']:
        if prev_ts is not None:
            dt = ts - prev_ts
            if ts is pd.NaT:
                counter=0
            elif dt == time_step:
                counter+=1
                if (counter == thresholdglucose):
                    counter = 0
                    count += 1
            elif dt < time_step or \
                    (time_step < dt < (time_step * 1.5)):

                if dt < (time_step - pd.Timedelta(seconds=10)):
                    counter = 0

                elif dt < time_step:
                    counter += 1
                    if (counter == thresholdglucose):
                        counter = 0
                        count += 1
                elif time_step < dt <= (time_step + pd.Timedelta(seconds=1)):
                    counter += 1
                    if (counter == thresholdglucose):
                        counter = 0
                        count += 1
                else:
                    counter=0
        else:
            counter+=1

        prev_ts = ts

    return count
		
