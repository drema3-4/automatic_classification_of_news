def print_coherence_by_epochs(self) -> None:
    plt.plot(
        range(len(self.coherence_by_epoch)),
        self.coherence_by_epoch,
        label="coherence"
    )
    plt.title("������ �������������")
    plt.xlabel("Epoch")
    plt.ylabel("Coherence")
    plt.legend()
    plt.show()