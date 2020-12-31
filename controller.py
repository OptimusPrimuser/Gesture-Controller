from Detection import detection
import pyautogui
import threading

#functions for controling mouse
def mouseMover(posX,posY):
    pyautogui.moveTo(posX,posY,duration=0)

def rightClick():
    pyautogui.rightClick()

def leftClick():
    pyautogui.rightClick()

pyautogui.FAILSAFE = False

#width and height of the screen
width,height=pyautogui.size()

#Activates or deactivates the mouse
is_activated=False
previous_Action=None
#actions
#"cursor", "fist", "Left Click", "Palm","Right Click"

for data in detection(0):

    #width ratio
    w_ratio=width/data['frameSize'][0]
    #height ratio
    h_ratio=height/data['frameSize'][1]
    
    posX=int(w_ratio*data['postion'][0])
    posY=int(w_ratio*data['postion'][1])

    #activate
    if data['Action']=="Palm":
        is_activated=True
    
    #deactivate
    elif data['Action']=="fist":
        is_activated=False    
    
    #move cursor
    elif data['Action']=="cursor" and is_activated==True:
        threading.Thread(target=mouseMover,args=[posX,posY]).start()
    
    #right click
    elif data['Action']=="Right Click" and is_activated==True and previous_Action!=data['Action']:
        threading.Thread(target=rightClick).start()
    
    #left click
    elif data['Action']=="Left Click" and is_activated==True and previous_Action!=data['Action']:
        threading.Thread(target=leftClick).start()
    
    previous_Action=data['Action']
    #print(is_activated)