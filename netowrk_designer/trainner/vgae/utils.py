from pytorch_metric_learning import losses, miners, testers

def get_all_embeddings(dataset, model):
    model.eval()
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)