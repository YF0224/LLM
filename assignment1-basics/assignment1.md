# 2.BPE

## 2.1 The Unicode Standard

* a '\x00' " "

* b __repr__() 会把“看不见的字符”如实暴露出来，而 print 打印的是“真实字符本身”，哪怕它根本显示不出来。

* c 

```python
    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    >>> "this is a test" + chr(0) + "string"
    'this is a test\x00string'
    >>> print("this is a test" + chr(0) + "string")
    this is a teststring
    >>>
```

## 2.2 Unicode Encodings

* a 更偏好在 UTF-8 字节上训练 tokenizer，因为 UTF-8 是变长、向后兼容 ASCII、且对英文/代码极其高效，而 UTF-16/32 要么浪费大量空间、要么引入端序和代理对复杂性，不利于稳定、紧凑的子词建模。

* b 该函数逐字节解码 UTF-8，而 UTF-8 的多字节字符必须整体解码，单独解码每个字节会导致解码错误或错误字符。

* c 这是一个非法的 UTF-8 过长编码（overlong encoding），不对应任何合法的 Unicode 字符，因此无法解码。

## 2.4 BPE Tokenizer Training
