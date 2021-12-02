def base32(content,length=8):
    if(not isinstance(content,int)):
        return base32(hashi(content,length=length*5),length=length)
    ch='0123456789abcdefghijklmnopqrstuvwxyz'
    ret=[]
    
    mask=(1<<(length*5))-1
    while(content>>(length*5)):
        content=(content>>(length*5))^(content&mask)
    mask=0b11111

    for i in range(length):
        ret.append(ch[content&mask])
        content>>=5
    return ''.join(ret[::-1])
def hashi(s,*args,**kwargs):
    if(isinstance(s,str)):
        return shashi(s,*args,**kwargs)
    elif(isinstance(s,int)):
        return s
    elif(isinstance(s,_io.BufferedReader)):
        return readerhashi(s,*args,**kwargs)
    else:
        return TypeError(type(s))
def shashi(s,encoding='utf-8',length=64):
    bytes=s.encode(encoding)
    ret=0
    for i in bytes:
        ret=(ret<<7)|int(i)
    mask=(1<<length)-1
    while(ret>>length):
        ret=(ret&mask)^(ret>>length)
    return ret