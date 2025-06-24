# -*- coding: utf-8 -*-
class BaseModel:
    def __init__(self, **data):
        for field in self.__annotations__:
            setattr(self, field, data.get(field))

    @classmethod
    def model_validate(cls, data):
        return cls(**data)

    def model_dump(self):
        """Return model data as a plain ``dict``."""
        return {field: getattr(self, field) for field in self.__annotations__}

    def model_dump_json(self, **kwargs):
        """Return model data serialized to JSON."""
        import json

        return json.dumps(self.model_dump(), **kwargs)
