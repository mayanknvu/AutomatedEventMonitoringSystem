import face_detection as fd
import train_model as tm
import monitor as mn
import sqlite3

def main():
    participantCount = 0
    conn = sqlite3.connect('FACE_DB.db')
    c = conn.cursor()
    names = []
    faces = []
    # Create table - CLIENTS
    c.execute('''CREATE TABLE Participants ([ID] integer PRIMARY KEY,[Name] text)''')
    while(True):
        print('1. Register Participant')
        print('2. Train Model')
        print('3. Start Monitoring')
        print('4. Quit')
        choice = int(input('\nEnter your choice: '))
        if choice == 1:
            participantCount = fd.RegisterFace(participantCount)
        elif choice == 2:
            names, faces = tm.TrainModel()
        elif choice == 3:
            mn.StartMonitoring(names, faces)
        elif choice == 4:
            break
        else:
            print('\nInvalid Choice! Try again!!')
            
main()