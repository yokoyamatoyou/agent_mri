class BaseModel:
    def __init__(self, **data):
        for field in self.__annotations__:
            setattr(self, field, data.get(field))

    @classmethod
    def model_validate(cls, data):
        return cls(**data)
