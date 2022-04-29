from pathlib import Path

def delete():
    for p in Path('.').glob('**/*.log'):
        print(p)
        p.unlink()
        

if __name__ == "__main__":
    delete()


