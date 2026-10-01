# delete_data.py — Safely delete collected images
import os
import shutil

DATA_FOLDER = "asl_data"

if not os.path.exists(DATA_FOLDER):
    print(f"No data folder found: {DATA_FOLDER}")
    input("Press Enter to exit...")
    exit()

print("Available letters:")
folders = [f for f in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, f))]
for i, folder in enumerate(folders):
    count = len(os.listdir(os.path.join(DATA_FOLDER, folder)))
    print(f"  {i+1}. {folder} → {count} images")

print("\nOptions:")
print("  0. Delete ALL data (full reset)")
for i, folder in enumerate(folders):
    print(f"  {i+1}. Delete only '{folder}'")
print("  Q. Quit without deleting")

choice = input("\nWhat do you want to delete? (0/1/2/... or Q): ").strip()

if choice.upper() == 'Q':
    print("Nothing deleted. Bye!")
    input("Press Enter to exit...")
    exit()

if choice == '0':
    confirm = input(f"\n⚠️  This will delete ALL images in '{DATA_FOLDER}' forever!\nType YES to confirm: ")
    if confirm == "YES":
        shutil.rmtree(DATA_FOLDER)
        os.makedirs(DATA_FOLDER)
        print("All data deleted and folder recreated.")
    else:
        print("Cancelled.")
else:
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(folders):
            letter = folders[idx]
            path = os.path.join(DATA_FOLDER, letter)
            confirm = input(f"\nDelete all images for letter '{letter}' ({len(os.listdir(path))} files)? Type YES: ")
            if confirm == "YES":
                shutil.rmtree(path)
                print(f"Deleted folder: {letter}")
            else:
                print("Cancelled.")
        else:
            print("Invalid number!")
    except:
        print("Invalid input!")

input("\nPress Enter to close...")