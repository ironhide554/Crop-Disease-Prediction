"""
Crop Disease & Pest Prediction Package
"""
__version__ = "1.0.0"

# Seamless compatibility polyfill for protobuf 5/6 and TensorFlow 2.14
try:
    import google.protobuf.message_factory
    if not hasattr(google.protobuf.message_factory.MessageFactory, 'GetPrototype'):
        from google.protobuf import message_factory
        def _get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)
        google.protobuf.message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass
