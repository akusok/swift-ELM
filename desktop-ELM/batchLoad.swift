
import Foundation
import Accelerate
import MetalPerformanceShaders

public typealias Float16 = UInt16

/*
 Uses vImage to convert a buffer of float16 values to regular Swift Floats.
 */
public func float16to32(_ input: UnsafeMutableRawPointer, count: Int) -> [Float] {
    var output = [Float](repeating: 0, count: count)
    var bufferFloat16 = vImage_Buffer(data: input,   height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: &output, height: 1, width: UInt(count), rowBytes: count * 4)
    
    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
        print("Error converting float16 to float32")
    }
    return output
}

func loadStuffInBatches() {

    let cq = device.makeCommandQueue()!
    var cbuf = cq.makeCommandBuffer()!


    // load input data
    let fileM = URL(fileURLWithPath: "/Users/akusok/wrkdir/b.npy")
    let m = loadNpyToMatrix(contentsOf: fileM)
    print("M", m.debugDescription)

    // load receiver matrix
    let fileR = URL(fileURLWithPath: "/Users/akusok/wrkdir/c.npy")
    let n = loadNpyToMatrix(contentsOf: fileR)



    // prepare batch matrix storage
    let batchRows = 16000
    let columns = m.columns
    let dt = m.dataType
    let rowBytes = MPSMatrixDescriptor.rowBytes(fromColumns: columns, dataType: dt)
    let rDescr = MPSMatrixDescriptor(rows: batchRows, columns: columns, rowBytes: rowBytes, dataType: dt)
    let r = MPSTemporaryMatrix(device: device, descriptor: rDescr)


    // prepare data load operation
    let xMean = Float(0.0)
    let xStd = Float(1.0)
    let matCopyNorm = MPSMatrixNeuron(device: device)
    matCopyNorm.setNeuronType(.linear, parameterA: 1.0/xStd, parameterB: -xMean, parameterC: 0.0)



    // copy receiver matrix - simulating previous results
    cbuf = cq.makeCommandBuffer()!
    matCopyNorm.encode(commandBuffer: cbuf, inputMatrix: n, biasVector: nil, resultMatrix: r)
    cbuf.commit()


    let t0 = CFAbsoluteTimeGetCurrent()

    // copy + normalize data
    for j in 0..<1 {
        
        // reset batch storage matrix if needed
        if m.rows != batchRows {
            print("Updating matrix at step \(j).")
            cbuf = cq.makeCommandBuffer()!
            let blit = cbuf.makeBlitCommandEncoder()!
            blit.fill(buffer: r.data, range: 0..<r.data.length, value: 0)
            blit.endEncoding()
            cbuf.commit()
        }
        
        cbuf = cq.makeCommandBuffer()!
        matCopyNorm.encode(commandBuffer: cbuf, inputMatrix: m, biasVector: nil, resultMatrix: r)
        cbuf.commit()
    }

    cbuf.waitUntilCompleted()
    let t1 = CFAbsoluteTimeGetCurrent() - t0
    print("Runtime is \(t1)")


    let uDescr = MPSMatrixDescriptor(rows: m.columns, columns: m.columns, rowBytes: rowBytes, dataType: dt)
    let u = MPSTemporaryMatrix(device: device, descriptor: uDescr)
    let matMult = MPSMatrixMultiplication(device: device, transposeLeft: true, transposeRight: false, resultRows: m.columns, resultColumns: m.columns, interiorColumns: m.rows, alpha: 1.0, beta: 0.0)


    let t2 = CFAbsoluteTimeGetCurrent()

    cbuf = cq.makeCommandBuffer()!
    matMult.encode(commandBuffer: cbuf, leftMatrix: r, rightMatrix: r, resultMatrix: u)
    cbuf.commit()
    cbuf.waitUntilCompleted()

    let t3 = CFAbsoluteTimeGetCurrent() - t2
    print("Runtime is \(t3)")


    // get results back
    let u2 = MPSMatrix(device: device, descriptor: uDescr)

    let matCopyBack = MPSMatrixCopy(device: device, copyRows: n.columns, copyColumns: n.columns, sourcesAreTransposed: false, destinationsAreTransposed: false)
    let zeroCopyOffset = MPSMatrixCopyOffsets(sourceRowOffset: 0, sourceColumnOffset: 0, destinationRowOffset: 0, destinationColumnOffset: 0)
    let matCopyDescr = MPSMatrixCopyDescriptor(sourceMatrix: u, destinationMatrix: u2, offsets: zeroCopyOffset)

    cbuf = cq.makeCommandBuffer()!
    matCopyBack.encode(commandBuffer: cbuf, copyDescriptor: matCopyDescr)
    cbuf.commit()

    cbuf = cq.makeCommandBuffer()!
    let blit = cbuf.makeBlitCommandEncoder()!
    blit.synchronize(resource: u2.data)
    blit.endEncoding()
    cbuf.commit()
    cbuf.waitUntilCompleted()


    if dt == MPSDataType.float16 {
        let A2 = float16to32(u2.data.contents(), count: 10)
        print(A2[0..<3])
    } else {
        let V2 = UnsafeMutableBufferPointer(start: u2.data.contents().assumingMemoryBound(to: Float.self), count: 10)
        print(Array(V2)[0..<3])
    }


    //
    //// get results back
    //let matCopyBack = MPSMatrixCopy(device: device, copyRows: n.rows, copyColumns: n.columns, sourcesAreTransposed: false, destinationsAreTransposed: false)
    //let zeroCopyOffset = MPSMatrixCopyOffsets(sourceRowOffset: 0, sourceColumnOffset: 0, destinationRowOffset: 0, destinationColumnOffset: 0)
    //let matCopyDescr = MPSMatrixCopyDescriptor(sourceMatrix: r, destinationMatrix: n, offsets: zeroCopyOffset)
    //
    //cbuf = cq.makeCommandBuffer()!
    //matCopyBack.encode(commandBuffer: cbuf, copyDescriptor: matCopyDescr)
    //cbuf.commit()
    //cbuf.waitUntilCompleted()
    //
    //
    //let Vp2 = n.data.contents().assumingMemoryBound(to: UInt16.self)
    //let V2 = UnsafeMutableBufferPointer(start: Vp2, count: n.rows * n.columns)
    //let A2 = Array(V2)
    //
    //print(stride(from: 0, to: A2.count, by: n.columns * 100).map { A2[$0] })



    print("Done here!")
    sleep(20)


    //let X: MPSMatrix = loadFromNpy(contentsOf: fileX)
    //let Y: MPSMatrix = loadFromNpy(contentsOf: fileY)
    //
    //let t0 = CFAbsoluteTimeGetCurrent()
    //
    //let model = ELM(bK: bK, bL: bL, alpha: 1E3, W: filesW, bias: filesBias)!
    //model.fit(X: X, Y: Y)
    //let Yh = model.predict(X: X)!
    //
    //let t = CFAbsoluteTimeGetCurrent() - t0
    //print("Solution took \(t) seconds")
    //
    //let res = Yh.data.contents().bindMemory(to: Float.self, capacity: Yh.rows * c)
    //for i in 0 ..< 10 {
    //    print(Array(0 ..< c).map { res[c*i + $0] })
    //}

}


