# Copyright (C) 2023 KNS Group LLC (YADRO)
# SPDX-License-Identifier: Apache-2.0


class UnregisteredProviderException(ValueError):
    def __init__(self, provider, root_provider):
        self.provider = provider
        self.root_provider = root_provider
        self.message = 'Requested provider {} not registered for {}'.format(provider, root_provider)
        super().__init__(self.message)


class BaseProvider:
    providers = {}
    __provider_type__ = None
    __provider__ = None

    @classmethod
    def provide(cls, provider, *args, **kwargs):
        root_provider = cls.resolve(provider)
        return root_provider(*args, **kwargs)

    @classmethod
    def resolve(cls, name):
        if name not in cls.providers:
            raise UnregisteredProviderException(name, cls.__provider_type__)
        return cls.providers[name]


class ClassProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if '_is_base_provider' in attrs:
            return cls

        assert issubclass(cls, ClassProvider), "Do not use metaclass directly"
        if '__provider_type__' in attrs:
            cls.providers = {}
        else:
            cls.register_provider(cls)

        return cls


class ClassProvider(BaseProvider, metaclass=ClassProviderMeta):
    _is_base_provider = True

    @classmethod
    def get_provider_name(cls):
        return getattr(cls, '__provider__', cls.__name__)

    @classmethod
    def register_provider(cls, provider):
        provider_name = cls.get_provider_name()
        if not provider_name:
            return
        cls.providers[provider_name] = provider
