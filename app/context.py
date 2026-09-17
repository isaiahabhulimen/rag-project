class AppContext:
    def __init__(
        self,
        text_collection,
        image_collection,
        ids,
        documents,
        metadatas,
        model,
        cross_encoder,
    ):
        self.text_collection = text_collection
        self.image_collection = image_collection
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.model = model
        self.cross_encoder = cross_encoder
