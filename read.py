import time

enable_obj = False
obj_timer = 0

while True:
    while enable_obj is False:
        obj_timer = time.time()
        enable_obj = True  # Resetting enable_obj to None after setting obj_timer
    if time.time() - obj_timer >= 1:  # Checking if 1 second has elapsed since obj_timer was last set
        # print("swap thread")
        print(time.time() - obj_timer)
        enable_obj = False