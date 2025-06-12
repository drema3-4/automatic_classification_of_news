def __make_vowpal_wabbit__(self) -> None:
    f = open(self.path_vw, "w")
    for row in range(self.data.shape[0]):
        string = ""
        for column in self.data.columns:
            string += str(self.data.loc[row, column]) + " "
        f.write("doc_{0} ".format(row) + string.strip() + "\n")