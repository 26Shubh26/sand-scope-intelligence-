import matplotlib.pyplot as plt

train_loss = [0.62, 0.43, 0.40, 0.37, 0.36, 0.34, 0.34, 0.34, 0.33, 0.34,
              0.32, 0.32, 0.32, 0.31, 0.33, 0.31, 0.31, 0.31, 0.31, 0.31]

val_loss = [0.50, 0.45, 0.41, 0.39, 0.39, 0.40, 0.39, 0.38, 0.37, 0.37,
            0.36, 0.35, 0.35, 0.37, 0.35, 0.35, 0.35, 0.34, 0.34, 0.34]

plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()