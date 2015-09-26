class LogColors:
    def __init__(self):
        self.HEADER = ''
        self.INFO = ''
        self.OK = ''
        self.WARN = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''
        self.UNDERLINE = ''

    def enable(self):
        self.HEADER = '\033[95m'
        self.INFO = '\033[94m'
        self.OK = '\033[92m'
        self.WARN = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'

    def disable(self):
        self.HEADER = ''
        self.INFO = ''
        self.OK = ''
        self.WARN = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''
        self.UNDERLINE = ''

# if __name__ == "__main__":
#     objName = bcolors()
#     objName.main() 