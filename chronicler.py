# food chronicler aka diet journal
import datetime

class Journal:
    def __init__(self, filename):
        self.today = datetime.date.today()

        if filename != "":
            self.filename = filename
        else:
            # default to today's date
            self.filename =  str(today) + ".txt"

        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)

    def save(self):
        with open(self.filename, "w") as f:
            for entry in self.entries:
                f.write(entry + "\n")

    def load(self):
        with open(self.filename, "r") as f:
            for line in f:
                self.entries.append(line.rstrip())

    def print_entries(self):
        for entry in self.entries:
            print(entry)


def main():
    # Create a new journal
    journal = Journal()

    # Add some entries
    journal.add_entry("First entry.")
    journal.add_entry("Second entry.")

    # Save the journal
    journal.save()

    # Print the entries
    journal.print_entries()

main()