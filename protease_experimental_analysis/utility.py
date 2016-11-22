import traitlets
from traitlets import TraitError

import inspect

def _all_subclasses(klass):
    direct_subclasses = klass.__subclasses__()
    sub_subclasses = [_all_subclasses(sc) for sc in direct_subclasses]

    return [
        sc 
        for clist in [direct_subclasses] + sub_subclasses
        for sc in clist
    ]

def resolve_subclass(baseclass, subclass):
    if isinstance(subclass, basestring):
        subclasses = _all_subclasses(baseclass)
        subclass_matches = [
            sc for sc in subclasses
            if ("%s.%s" % (sc.__module__, sc.__name__)) == subclass
            or sc.__name__ == subclass
        ]
        if len(subclass_matches) == 1:
            return subclass_matches[0]
        elif len(subclass_matches) > 1:
            raise TraitError("Ambiguous subclass name: %s subclasses: %s" % (subclass, subclass_matches))
        else:
            raise TraitError("No matching subclass name: %s" % subclass)
    else:
        if not issubclass(subclass, baseclass):
            raise TraitError("Specific class not subclass: %s base: %s" % (subclass, baseclass))
            
        return subclass

class SubclassName(traitlets.DottedObjectName):
    def __init__(self, klass, **kwargs):
        if not inspect.isclass(klass):
            raise TraitError('The klass attribute must be a class not: %r' % klass)

        self.klass = klass
        
        super(SubclassName, self).__init__(**kwargs)

    def validate(self, obj, value):
        value = super(SubclassName, self).validate(obj, value)

        resolve_subclass(self.klass, value)
        
        return value
